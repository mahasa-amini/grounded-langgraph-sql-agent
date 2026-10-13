from typing import Any

from tools.aggregation_detection_tool import detect_aggregation_from_phrase
from tools.schema_dimension_tool import normalize_schema_phrase


_AGGREGATION_PREFIXES = {
    "avg": ("average", "avg", "mean"),
    "max": ("maximum", "max", "highest"),
    "min": ("minimum", "min", "lowest"),
    "sum": ("total", "sum"),
    "count": ("count",),
}


def _queryable_metric_terms(public_schema_context: dict[str, Any]) -> set[str]:
    queryable_metrics = public_schema_context.get("queryable_metrics", [])
    if isinstance(queryable_metrics, dict):
        items = [
            {"concept": key, **(value if isinstance(value, dict) else {})}
            for key, value in queryable_metrics.items()
        ]
    elif isinstance(queryable_metrics, list):
        items = [item for item in queryable_metrics if isinstance(item, dict)]
    else:
        items = []

    terms: set[str] = set()
    for item in items:
        for value in (
            item.get("concept"),
            item.get("key"),
            item.get("name"),
            item.get("label"),
            *(item.get("synonyms", []) or []),
        ):
            if value:
                terms.add(normalize_schema_phrase(value))
    return terms


def normalize_metric_aggregation_evidence_tool(
    metric_phrase: str | None,
    aggregation_phrase: str | None,
    public_schema_context: dict[str, Any],
    requirements_context: dict[str, Any],
) -> dict[str, Any]:
    """Normalize an exact ``aggregation + supported metric`` phrase.

    The metric remainder must be an exact public-schema alias. This deliberately
    does not turn arbitrary phrases such as ``average pressure`` into supported
    metrics merely because an LLM selected the generic ``value`` column.
    """
    normalized_metric = normalize_schema_phrase(metric_phrase) if metric_phrase else ""
    explicit_aggregation = detect_aggregation_from_phrase(aggregation_phrase)
    supported_aggregations = set(
        requirements_context.get("aggregations", {}).get("supported", []) or []
    )
    metric_terms = _queryable_metric_terms(public_schema_context)

    compound_aggregation = None
    normalized_metric_phrase = metric_phrase.strip() if isinstance(metric_phrase, str) else None
    for aggregation, prefixes in _AGGREGATION_PREFIXES.items():
        if aggregation not in supported_aggregations:
            continue
        for prefix in prefixes:
            normalized_prefix = normalize_schema_phrase(prefix)
            marker = f"{normalized_prefix} "
            if normalized_metric.startswith(marker):
                remainder = normalized_metric[len(marker):].strip()
                if remainder in metric_terms:
                    compound_aggregation = aggregation
                    normalized_metric_phrase = remainder
                    break
        if compound_aggregation:
            break

    conflict = None
    if (
        compound_aggregation
        and explicit_aggregation
        and compound_aggregation != explicit_aggregation
    ):
        conflict = (
            "Conflicting aggregation evidence: metric phrase implies "
            f"{compound_aggregation}, but aggregation phrase implies "
            f"{explicit_aggregation}."
        )

    effective_aggregation = explicit_aggregation or compound_aggregation
    return {
        "metric_phrase": normalized_metric_phrase,
        "aggregation_phrase": effective_aggregation or aggregation_phrase,
        "compound_aggregation": compound_aggregation,
        "explicit_aggregation": explicit_aggregation,
        "aggregation_conflict": conflict,
    }
