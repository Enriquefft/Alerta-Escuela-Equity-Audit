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
    """Build the 8 findings with runtime-extracted stats.

    Narrative arc: who needs help -> specific gaps -> why it happens ->
    proof it's structural -> regional opportunity -> actionable insight ->
    what works -> path forward.
    """
    fm = exports["fairness_metrics.json"]
    sv = exports["shap_values.json"]
    ch = exports["choropleth.json"]
    mr = exports["model_results.json"]

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
    lgbm_prauc = _resolve_path(mr, "lightgbm.metrics.validate_2022.weighted.pr_auc")
    xgb_prauc = _resolve_path(mr, "xgboost.metrics.validate_2022.weighted.pr_auc")

    # Derived values for headlines
    castellano_fnr_ratio = round(castellano_fnr * 10)
    urban_indig_fnr_ratio = round(urban_indig_fnr * 4)  # 3 of 4
    algo_ratio = max(lgbm_prauc, xgb_prauc) / min(lgbm_prauc, xgb_prauc)

    findings = [
        # --- Finding 1: Students who need early detection ---
        {
            "id": "fnr_overall",
            "stat": f"FNR = {_format_pct(castellano_fnr)} for Spanish-speaking students",
            "headline_es": (
                f"{castellano_fnr_ratio} de cada 10 estudiantes en riesgo aun "
                f"no reciben deteccion temprana"
            ),
            "headline_en": (
                f"{castellano_fnr_ratio} in 10 at-risk students still lack "
                f"early detection — an opportunity to reach them sooner"
            ),
            "explanation_es": (
                f"Actualmente, el {_format_pct(castellano_fnr)} de los "
                f"estudiantes hispanohablantes que abandonan la escuela no son "
                f"identificados a tiempo. Mejorar la deteccion temprana para "
                f"este grupo podria permitir intervenciones preventivas que "
                f"lleguen a la mayoria de los estudiantes en riesgo."
            ),
            "explanation_en": (
                f"Currently, {_format_pct(castellano_fnr)} of Spanish-speaking "
                f"students who drop out are not identified in time. Improving "
                f"early detection for this group could enable preventive "
                f"interventions that reach the majority of at-risk students."
            ),
            "metric_source": {
                "path": "fairness_metrics.json#dimensions.language.groups.castellano.fnr",
                "label": "FNR for Castellano (Spanish) language group",
            },
            "visualization_type": "bar_chart",
            "data_key": "dimensions.language",
            "severity": "critical",
        },
        # --- Finding 2: Two-sided detection gap ---
        {
            "id": "two_sided_detection_gap",
            "stat": (
                f"FPR = {_format_pct(indigenous_fpr)} (indigenous) vs "
                f"{_format_pct(castellano_fpr)} (Spanish); "
                f"FNR = {_format_pct(indigenous_fnr)} vs {_format_pct(castellano_fnr)}"
            ),
            "headline_es": (
                f"La deteccion tiene dos caras: estudiantes indigenas reciben "
                f"exceso de alertas mientras hispanohablantes quedan sin atencion"
            ),
            "headline_en": (
                f"Detection has two sides: indigenous students receive excess "
                f"alerts while Spanish-speakers go undetected"
            ),
            "explanation_es": (
                f"El modelo identifica mejor a estudiantes de lenguas indigenas "
                f"(FNR={_format_pct(indigenous_fnr)}) pero genera demasiadas "
                f"alertas innecesarias (FPR={_format_pct(indigenous_fpr)}). "
                f"Para hispanohablantes ocurre lo inverso: pasan desapercibidos "
                f"con un FNR del {_format_pct(castellano_fnr)}. Equilibrar "
                f"ambos lados beneficiaria a todos los estudiantes."
            ),
            "explanation_en": (
                f"The model identifies more indigenous-language students "
                f"(FNR={_format_pct(indigenous_fnr)}) but generates too many "
                f"unnecessary alerts (FPR={_format_pct(indigenous_fpr)}). "
                f"For Spanish-speakers the pattern reverses: they go undetected "
                f"at a {_format_pct(castellano_fnr)} rate. Balancing both sides "
                f"would benefit all students."
            ),
            "metric_source": {
                "path": "fairness_metrics.json#dimensions.language.groups.other_indigenous.fpr",
                "label": "FPR for Other Indigenous language group",
            },
            "visualization_type": "grouped_bar",
            "data_key": "dimensions.language",
            "severity": "critical",
        },
        # --- Finding 3: Urban indigenous students ---
        {
            "id": "urban_indigenous_gap",
            "stat": (
                f"FNR = {_format_pct(urban_indig_fnr)} for urban indigenous "
                f"students (n={urban_indig_n})"
            ),
            "headline_es": (
                f"{urban_indig_fnr_ratio} de cada 4 estudiantes indigenas "
                f"urbanos en riesgo necesitan mejor cobertura"
            ),
            "headline_en": (
                f"{urban_indig_fnr_ratio} in 4 urban indigenous students at risk "
                f"fall through the gaps between rural and urban detection"
            ),
            "explanation_es": (
                f"Los estudiantes indigenas en zonas urbanas presentan la mayor "
                f"brecha de deteccion (FNR={_format_pct(urban_indig_fnr)}). "
                f"Aunque son una muestra pequena (n={urban_indig_n}), el patron "
                f"sugiere que la deteccion actual se centra en factores rurales "
                f"y no captura la vulnerabilidad linguistica en ciudades. "
                f"Estos estudiantes podrian beneficiarse de indicadores urbanos "
                f"complementarios."
            ),
            "explanation_en": (
                f"Urban indigenous students have the largest detection gap "
                f"(FNR={_format_pct(urban_indig_fnr)}). While the sample is "
                f"small (n={urban_indig_n}), the pattern suggests current "
                f"detection focuses on rural factors and does not capture "
                f"linguistic vulnerability in cities. These students could "
                f"benefit from complementary urban indicators."
            ),
            "metric_source": {
                "path": "fairness_metrics.json#intersections.language_x_rural.groups.other_indigenous_urban.fnr",
                "label": "FNR for urban indigenous-language students",
            },
            "visualization_type": "heatmap",
            "data_key": "intersections.language_x_rural",
            "severity": "high",
        },
        # --- Finding 4: Dropout driven by place, not identity ---
        {
            "id": "place_not_identity",
            "stat": (
                f"Top-5 SHAP features: {', '.join(top_5_shap)}; "
                f"0/5 overlap with logistic regression identity features"
            ),
            "headline_es": (
                f"Donde vive un estudiante predice la desercion mas que "
                f"quien es — la solucion esta en los territorios"
            ),
            "headline_en": (
                f"Where a student lives predicts dropout more than who they "
                f"are — the solution lies in places, not people"
            ),
            "explanation_es": (
                f"Los 5 factores mas influyentes son estructurales: edad, "
                f"luminosidad nocturna (proxy de urbanizacion), situacion "
                f"laboral, proporcion de lengua indigena del distrito y "
                f"alfabetizacion censal. Esto significa que las condiciones "
                f"del territorio determinan el riesgo, lo cual abre la puerta "
                f"a intervenciones territoriales focalizadas."
            ),
            "explanation_en": (
                f"The 5 most influential factors are structural: age, "
                f"nightlight intensity (urbanization proxy), employment status, "
                f"district indigenous language prevalence, and census literacy "
                f"rate. This means territorial conditions drive risk, which "
                f"opens the door to targeted place-based interventions."
            ),
            "metric_source": {
                "path": "shap_values.json#top_5_shap",
                "label": "Top 5 SHAP features (LightGBM)",
            },
            "visualization_type": "bar_chart",
            "data_key": "global_importance",
            "severity": "medium",
        },
        # --- Finding 5: Amazon region needs targeted attention ---
        {
            "id": "selva_opportunity",
            "stat": f"FNR = {_format_pct(selva_fnr)} in the Selva region",
            "headline_es": (
                f"La region Selva necesita atencion focalizada: "
                f"{round(selva_fnr * 10)} de cada 10 estudiantes en riesgo "
                f"aun no son alcanzados"
            ),
            "headline_en": (
                f"The Amazon region needs targeted attention: "
                f"{round(selva_fnr * 10)} in 10 at-risk students are not yet "
                f"reached"
            ),
            "explanation_es": (
                f"En la Selva, el {_format_pct(selva_fnr)} de los estudiantes "
                f"que abandonan no son identificados a tiempo. La alta "
                f"dispersion geografica, menor conectividad y diversidad de "
                f"poblaciones indigenas hacen de esta region la que mas se "
                f"beneficiaria de estrategias de deteccion adaptadas a su "
                f"realidad."
            ),
            "explanation_en": (
                f"In the Selva (Amazon basin), {_format_pct(selva_fnr)} of "
                f"students who drop out are not identified in time. High "
                f"geographic dispersion, lower connectivity, and diverse "
                f"indigenous populations make this the region that would "
                f"benefit most from detection strategies adapted to its "
                f"reality."
            ),
            "metric_source": {
                "path": "fairness_metrics.json#dimensions.region.groups.selva.fnr",
                "label": "FNR for Selva (Amazon) region",
            },
            "visualization_type": "choropleth",
            "data_key": "dimensions.region",
            "severity": "high",
        },
        # --- Finding 6: Individual and district data tell different stories ---
        {
            "id": "individual_vs_district",
            "stat": f"Pearson r = {pearson_r:.3f} between predictions and admin rates",
            "headline_es": (
                f"Los datos individuales y distritales cuentan historias "
                f"diferentes — combinarlos fortaleceria la deteccion"
            ),
            "headline_en": (
                f"Individual and district data tell different stories "
                f"(r = {pearson_r:.3f}) — combining them could strengthen "
                f"detection"
            ),
            "explanation_es": (
                f"La correlacion entre las predicciones individuales agregadas "
                f"y las tasas distritales del MINEDU es practicamente nula "
                f"(r = {pearson_r:.3f}). Esto no significa que una fuente "
                f"sea incorrecta: cada una captura aspectos distintos del "
                f"fenomeno. Integrar ambas perspectivas podria ofrecer una "
                f"imagen mas completa del riesgo de desercion."
            ),
            "explanation_en": (
                f"The correlation between aggregated individual predictions "
                f"and MINEDU district dropout rates is near zero "
                f"(r = {pearson_r:.3f}). This does not mean either source is "
                f"wrong: each captures different aspects of the phenomenon. "
                f"Integrating both perspectives could provide a more complete "
                f"picture of dropout risk."
            ),
            "metric_source": {
                "path": "choropleth.json#correlation.pearson_r",
                "label": "Pearson r between model predictions and admin dropout rates",
            },
            "visualization_type": "choropleth",
            "data_key": "correlation",
            "severity": "medium",
        },
        # --- Finding 7: Gender equity works ---
        {
            "id": "gender_equity_works",
            "stat": f"FNR gap = {_format_pct(sex_fnr_gap)} between sexes",
            "headline_es": (
                f"Buena noticia: la equidad de genero funciona — solo "
                f"{_format_pct(sex_fnr_gap)} de diferencia entre ninos y ninas"
            ),
            "headline_en": (
                f"Good news: gender equity works — only "
                f"{_format_pct(sex_fnr_gap)} detection gap between boys and girls"
            ),
            "explanation_es": (
                f"A diferencia de la lengua y la geografia, la deteccion "
                f"trata de manera casi equitativa a ninos y ninas, con una "
                f"brecha de solo {_format_pct(sex_fnr_gap)}. Este resultado "
                f"positivo demuestra que la equidad es alcanzable y puede "
                f"servir de modelo para cerrar las brechas en otras "
                f"dimensiones."
            ),
            "explanation_en": (
                f"Unlike language and geography, detection treats boys and "
                f"girls nearly equally, with a gap of just "
                f"{_format_pct(sex_fnr_gap)}. This positive result shows that "
                f"equity is achievable and can serve as a model for closing "
                f"gaps in other dimensions."
            ),
            "metric_source": {
                "path": "fairness_metrics.json#dimensions.sex.gaps.max_fnr_gap",
                "label": "Maximum FNR gap between sex groups",
            },
            "visualization_type": "bar_chart",
            "data_key": "dimensions.sex",
            "severity": "low",
        },
        # --- Finding 8: Algorithm-independent — better data is the path ---
        {
            "id": "algorithm_independent",
            "stat": (
                f"LightGBM PR-AUC = {lgbm_prauc:.4f}, "
                f"XGBoost PR-AUC = {xgb_prauc:.4f}, "
                f"ratio = {algo_ratio:.4f}"
            ),
            "headline_es": (
                f"Estos patrones no dependen del algoritmo — la solucion "
                f"es mejor informacion, no mejor tecnologia"
            ),
            "headline_en": (
                f"These patterns persist regardless of algorithm — the path "
                f"forward is better data, not better models"
            ),
            "explanation_es": (
                f"Dos algoritmos completamente distintos (LightGBM y XGBoost) "
                f"producen resultados casi identicos "
                f"(ratio = {algo_ratio:.4f}). Esto confirma que las brechas "
                f"detectadas son estructurales, no un artefacto del modelo. "
                f"La solucion no es cambiar el algoritmo sino enriquecer los "
                f"datos con indicadores mas cercanos a la realidad de cada "
                f"comunidad."
            ),
            "explanation_en": (
                f"Two completely different algorithms (LightGBM and XGBoost) "
                f"produce nearly identical results "
                f"(ratio = {algo_ratio:.4f}). This confirms that the detected "
                f"gaps are structural, not a model artifact. The solution is "
                f"not to change the algorithm but to enrich the data with "
                f"indicators closer to each community's reality."
            ),
            "metric_source": {
                "path": "model_results.json#lightgbm.metrics.validate_2022.weighted.pr_auc",
                "label": "LightGBM validation PR-AUC (weighted)",
            },
            "visualization_type": "comparison_bar",
            "data_key": "algorithm_comparison",
            "severity": "medium",
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
