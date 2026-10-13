from typing import Any



def _to_string_list(value: object) -> list[str]:
    """
    Converts a raw iterable value into a list of non-empty strings.
    """
    if not isinstance(value, (list, tuple, set)):
        return []

    return [
        item
        for item in value
        if isinstance(item, str) and item.strip()
    ]


def _to_string_or_none(value: object) -> str | None:
    """
    Converts a raw value into a non-empty string, otherwise returns None.
    """
    if isinstance(value, str) and value.strip():
        return value

    return None


def _has_aggregation(value: object) -> bool:
    """
    Returns whether an aggregation function is present.
    """
    return isinstance(value, str) and bool(value.strip())


def _has_unsupported_terms(value: object) -> bool:
    """
    Returns whether unsupported terms are present.
    """
    return isinstance(value, list) and len(value) > 0


def _collect_equality_filter_columns(filters: object) -> set[str]:
    """
    Collects columns that have equality filters.

    These columns should not be kept in GROUP BY when aggregation is used,
    because a single-value equality filter already fixes the dimension.
    """
    if not isinstance(filters, list):
        return set()

    equality_filter_columns: set[str] = set()

    for filter_item in filters:
        if not isinstance(filter_item, dict):
            continue

        column = filter_item.get("column")
        operator = filter_item.get("operator")

        if operator == "=" and isinstance(column, str) and column.strip():
            equality_filter_columns.add(column)

    return equality_filter_columns


def resolve_query_projection(
    mapped_selected_columns: object,
    mapped_group_by: object,
    requested_group_by_columns: object,
    available_columns: set[str],
    aggregation: object,
    filters: object,
    multi_value_filter_columns: object,
    unsupported_terms: object,
    time_column: object,
    metric_column: object,
) -> tuple[list[str], list[str]]:
    """
    Resolves selected columns and group-by columns for the mapped query.

    Rules:
    - Start from mapper group_by.
    - Add explicitly requested group-by dimensions.
    - Add multi-value filter columns to group_by when aggregation is used.
    - Remove single-value equality filter columns from group_by when aggregation is used.
    - If unsupported terms exist, selected_columns must be empty.
    - If aggregation is used, selected_columns should contain only group_by columns.
    - If no aggregation is used and selected_columns is empty, default to time + metric.
    """
    group_by = _to_string_list(mapped_group_by)
    selected_columns_from_mapping = _to_string_list(mapped_selected_columns)
    requested_group_by_columns_list = _to_string_list(requested_group_by_columns)
    multi_value_filter_columns_list = _to_string_list(multi_value_filter_columns)

    has_aggregation = _has_aggregation(aggregation)

    for column in requested_group_by_columns_list:
        if column in available_columns and column not in group_by:
            group_by.append(column)

    if has_aggregation:
        for column in multi_value_filter_columns_list:
            if column in available_columns and column not in group_by:
                group_by.append(column)

    if has_aggregation:
        equality_filter_columns = _collect_equality_filter_columns(filters)

        group_by = [
            column
            for column in group_by
            if column not in equality_filter_columns
        ]

    if _has_unsupported_terms(unsupported_terms):
        selected_columns: list[str] = []
    elif has_aggregation:
        selected_columns = [
            column
            for column in selected_columns_from_mapping
            if column in group_by
        ]

        for column in group_by:
            if column not in selected_columns:
                selected_columns.append(column)
    else:
        selected_columns = selected_columns_from_mapping

        resolved_time_column = _to_string_or_none(time_column)
        resolved_metric_column = _to_string_or_none(metric_column)

        if not selected_columns and resolved_time_column and resolved_metric_column:
            selected_columns = [resolved_time_column, resolved_metric_column]

    return selected_columns, group_by


def apply_post_guardrail_projection_cleanup(
    mapping: dict[str, Any],
    available_columns: set[str],
    multi_value_filter_columns: object,
) -> dict[str, Any]:
    """
    Applies projection cleanup after request evidence guardrails.

    This is needed because guardrails can change filters, group_by, selected_columns,
    or aggregation. The cleanup keeps SELECT and GROUP BY consistent after those changes.

    Rules:
    - Only applies when aggregation exists.
    - Remove single-value equality filter columns from GROUP BY.
    - Add multi-value filter columns to GROUP BY.
    - Keep selected_columns aligned with GROUP BY.
    - Remove order_by when it points to a time column that is not grouped.
    """
    cleaned_mapping = dict(mapping)

    if not _has_aggregation(cleaned_mapping.get("aggregation")):
        return cleaned_mapping

    filters = cleaned_mapping.get("filters", [])
    equality_filter_columns = _collect_equality_filter_columns(filters)

    group_by = [
        column
        for column in _to_string_list(cleaned_mapping.get("group_by"))
        if column not in equality_filter_columns
    ]

    for column in _to_string_list(multi_value_filter_columns):
        if column in available_columns and column not in group_by:
            group_by.append(column)

    selected_columns = [
        column
        for column in _to_string_list(cleaned_mapping.get("selected_columns"))
        if column in group_by
    ]

    for column in group_by:
        if column not in selected_columns:
            selected_columns.append(column)

    if not group_by:
        selected_columns = []
        cleaned_mapping["order_by"] = None
    else:
        order_by = _to_string_or_none(cleaned_mapping.get("order_by"))
        time_column = _to_string_or_none(cleaned_mapping.get("time_column"))

        if order_by and time_column:
            order_by_parts = order_by.strip().split()
            order_by_column = order_by_parts[0] if order_by_parts else None

            if order_by_column == time_column and time_column not in group_by:
                cleaned_mapping["order_by"] = None

    cleaned_mapping["group_by"] = group_by
    cleaned_mapping["selected_columns"] = selected_columns

    return cleaned_mapping