from typing import Any

from agent.state import QueryFilter
from tools.filter_normalization_tool import (
    deduplicate_filters,
    normalize_multi_value_equality_filters,
)
from tools.normalize_location_tool import normalize_location_tool


def resolve_query_filters(
    mapped_filters: object,
    previous_filters: object,
    mapping_input_text: str,
    semantic_context: object,
) -> tuple[list[QueryFilter], list[str]]:
    """
    Resolves query filters from previous state, mapper output, and location normalization.

    Rules:
    - Start with previous state filters.
    - Add filters produced by the schema mapper.
    - Add deterministic location filters based on the request text and semantic context.
    - Deduplicate filters.
    - Convert repeated equality filters on the same column into IN filters.
    """
    filters: list[QueryFilter] = []

    filters.extend(_normalize_previous_filters(previous_filters))
    filters.extend(_normalize_mapped_filters(mapped_filters))

    location_filters = normalize_location_tool(
        text=mapping_input_text,
        semantic_context=_normalize_semantic_context(semantic_context),
    )

    filters.extend(location_filters)
    filters = deduplicate_filters(filters)

    filters, multi_value_filter_columns = normalize_multi_value_equality_filters(filters)

    return filters, multi_value_filter_columns


def _normalize_previous_filters(previous_filters: object) -> list[QueryFilter]:
    """
    Normalizes filters already stored in state.
    """
    if not isinstance(previous_filters, list):
        return []

    normalized_filters: list[QueryFilter] = []

    for filter_item in previous_filters:
        if not isinstance(filter_item, dict):
            continue

        normalized_filter = _normalize_filter_dict(filter_item)

        if normalized_filter is not None:
            normalized_filters.append(normalized_filter)

    return normalized_filters


def _normalize_mapped_filters(mapped_filters: object) -> list[QueryFilter]:
    """
    Normalizes filters returned by the schema mapper.

    Mapper filters are usually pydantic objects with column/operator/value fields.
    """
    if not isinstance(mapped_filters, list):
        return []

    normalized_filters: list[QueryFilter] = []

    for filter_item in mapped_filters:
        column = getattr(filter_item, "column", None)
        operator = getattr(filter_item, "operator", None)
        value = getattr(filter_item, "value", None)

        normalized_filter = _normalize_filter_dict(
            {
                "column": column,
                "operator": operator,
                "value": value,
            }
        )

        if normalized_filter is not None:
            normalized_filters.append(normalized_filter)

    return normalized_filters


def _normalize_filter_dict(filter_item: dict[str, Any]) -> QueryFilter | None:
    """
    Converts a raw filter dict into QueryFilter when column/operator are valid.
    """
    column = filter_item.get("column")
    operator = filter_item.get("operator")

    if not isinstance(column, str) or not column.strip():
        return None

    if not isinstance(operator, str) or not operator.strip():
        return None

    return {
        "column": column,
        "operator": operator,
        "value": filter_item.get("value"),
    }


def _normalize_semantic_context(semantic_context: object) -> dict[str, Any]:
    """
    Normalizes semantic context before passing it to location normalization.
    """
    if isinstance(semantic_context, dict):
        return semantic_context

    return {}