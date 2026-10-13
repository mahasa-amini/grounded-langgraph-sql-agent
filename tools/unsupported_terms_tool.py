from typing import Any


KNOWN_UNSUPPORTED_METRIC_TERMS = {
    "pressure",
    "temperature",
    "level",
    "volume",
    "energy",
    "power",
    "humidity",
    "speed",
}


def detect_unsupported_terms_tool(
    text: str | None,
    public_schema_context: dict[str, Any],
) -> list[str]:
    if not text:
        return []

    normalized_text = text.lower()

    supported_metric_synonyms = _get_supported_metric_terms(public_schema_context)

    unsupported_terms: list[str] = []

    for term in KNOWN_UNSUPPORTED_METRIC_TERMS:
        if term in normalized_text and term not in supported_metric_synonyms:
            unsupported_terms.append(term)

    return unsupported_terms


def _get_supported_metric_terms(public_schema_context: dict[str, Any]) -> set[str]:
    concepts = public_schema_context.get("user_facing_concepts", {})
    flow_concept = concepts.get("flow", {})

    supported_terms = set()

    label = flow_concept.get("label")
    if label:
        supported_terms.add(str(label).lower())

    synonyms = flow_concept.get("synonyms", [])
    for synonym in synonyms:
        supported_terms.add(str(synonym).lower())

    return supported_terms