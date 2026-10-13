from typing import Any

from tools.ranking_cleanup_tool import normalize_ranking_text
from tools.schema_dimension_tool import (
    build_schema_dimension_aliases,
    collect_schema_dimension_columns_from_mapping,
)
from tools.semantic_mapping_tool import semantic_alias_exists_in_request_text


def term_cleanup_normalize(value: Any) -> str:
    """Normalize a term for safe request-text comparison."""
    return normalize_ranking_text(str(value))


def term_exists_in_request_text(
    term: Any,
    request_text: str,
) -> bool:
    """
    Return True only if the term really appears in the user request text.

    snake_case terms must match raw snake_case only.
    Example:
        flow_in should match "flow_in"
        flow_in should NOT match "flow in Marconi"
    """
    return semantic_alias_exists_in_request_text(
        alias=term,
        request_text=request_text,
    )


def iter_cleanup_values(value: Any) -> list[Any]:
    """Iterate scalar or list-like values used in filters."""
    if value is None:
        return []

    if isinstance(value, str):
        stripped_value = value.strip()
        return [stripped_value] if stripped_value else []

    if isinstance(value, (list, tuple, set)):
        values: list[Any] = []

        for item in value:
            if item is None:
                continue

            if isinstance(item, str):
                stripped_item = item.strip()

                if not stripped_item:
                    continue

                values.append(stripped_item)
                continue

            values.append(item)

        return values

    return [value]


def collect_resolved_terms_for_term_cleanup(
    mapping: dict[str, Any],
    request_evidence: dict[str, Any] | None,
    available_columns: set[str] | None = None,
) -> set[str]:
    """
    Collect terms that are already safely resolved by deterministic mapping.
    """
    resolved_terms: set[str] = set()

    for filter_item in mapping.get("filters", []) or []:
        for value in iter_cleanup_values(filter_item.get("value")):
            normalized_value = term_cleanup_normalize(value)

            if normalized_value:
                resolved_terms.add(normalized_value)

    if mapping.get("metric_column"):
        raw_metric = mapping.get("raw_metric")

        if raw_metric:
            normalized_raw_metric = term_cleanup_normalize(raw_metric)

            if normalized_raw_metric:
                resolved_terms.add(normalized_raw_metric)

        if request_evidence:
            metric_phrase = request_evidence.get("metric_phrase")

            if metric_phrase:
                normalized_metric_phrase = term_cleanup_normalize(metric_phrase)

                if normalized_metric_phrase:
                    resolved_terms.add(normalized_metric_phrase)

    dimension_columns = collect_schema_dimension_columns_from_mapping(
        mapping=mapping,
        available_columns=available_columns,
    )

    dimension_aliases = build_schema_dimension_aliases(dimension_columns)

    for alias in dimension_aliases:
        normalized_alias = term_cleanup_normalize(alias)

        if normalized_alias:
            resolved_terms.add(normalized_alias)

    return resolved_terms


def remove_hallucinated_or_resolved_terms(
    terms: list[Any],
    request_text: str,
    resolved_terms: set[str],
) -> list[Any]:
    """
    Remove terms that should not block validation.

    A term is removed if:
    - it does not actually appear in the user request text
    - or it has already been resolved deterministically
    """
    cleaned_terms: list[Any] = []
    seen_terms: set[str] = set()

    for term in terms or []:
        normalized_term = term_cleanup_normalize(term)

        if not normalized_term:
            continue

        if normalized_term in seen_terms:
            continue

        if not term_exists_in_request_text(
            term=term,
            request_text=request_text,
        ):
            continue

        if normalized_term in resolved_terms:
            continue

        seen_terms.add(normalized_term)
        cleaned_terms.append(term)

    return cleaned_terms