from tools.schema_dimension_tool import normalize_schema_phrase


def detect_aggregation_from_phrase(aggregation_phrase: str | None) -> str | None:
    """
    Detects the canonical aggregation function from a user-facing phrase.
    """
    if not aggregation_phrase:
        return None

    normalized_aggregation_phrase = normalize_schema_phrase(aggregation_phrase)

    if any(term in normalized_aggregation_phrase for term in ["average", "avg", "mean"]):
        return "avg"

    if any(term in normalized_aggregation_phrase for term in ["maximum", "max", "highest"]):
        return "max"

    if any(term in normalized_aggregation_phrase for term in ["minimum", "min", "lowest"]):
        return "min"

    if any(term in normalized_aggregation_phrase for term in ["total", "sum"]):
        return "sum"

    if "count" in normalized_aggregation_phrase:
        return "count"

    return None