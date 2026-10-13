from typing import cast

from agent.state import ActiveRequestType, ChartType, OutputType


_VALID_ACTIVE_REQUEST_TYPES = {"data", "chart"}
_VALID_OUTPUT_TYPES = {"text", "chart"}
_VALID_CHART_TYPES = {"line", "bar", "table"}


def _to_active_request_type(value: object) -> ActiveRequestType | None:
    """
    Converts a raw value into a valid ActiveRequestType.
    """
    if value in _VALID_ACTIVE_REQUEST_TYPES:
        return cast(ActiveRequestType, value)

    return None


def _to_output_type(value: object) -> OutputType | None:
    """
    Converts a raw value into a valid OutputType.
    """
    if value in _VALID_OUTPUT_TYPES:
        return cast(OutputType, value)

    return None


def _to_chart_type(value: object) -> ChartType | None:
    """
    Converts a raw value into a valid ChartType.
    """
    if value in _VALID_CHART_TYPES:
        return cast(ChartType, value)

    return None


def resolve_output_fields(
    mapped_active_request_type: object,
    previous_active_request_type: object,
    mapped_output_type: object,
    previous_output_type: object,
    mapped_chart_type: object,
    previous_chart_type: object,
) -> tuple[ActiveRequestType | None, OutputType | None, ChartType | None]:
    """
    Resolves active request type, output type, and chart type.

    Rules:
    - Prefer mapped values over previous state values.
    - If the active request is chart, force output_type to chart.
    - If the active request is data, force output_type to text.
    - If output_type is chart and chart_type is missing, default to line.
    """
    active_request_type = (
        _to_active_request_type(mapped_active_request_type)
        or _to_active_request_type(previous_active_request_type)
    )

    output_type = (
        _to_output_type(mapped_output_type)
        or _to_output_type(previous_output_type)
    )

    if active_request_type == "chart":
        output_type = "chart"

    if active_request_type == "data":
        output_type = "text"

    chart_type = (
        _to_chart_type(mapped_chart_type)
        or _to_chart_type(previous_chart_type)
    )

    if output_type == "chart" and chart_type is None:
        chart_type = "line"

    return active_request_type, output_type, chart_type

def _to_string_or_none(value: object) -> str | None:
    """
    Converts a raw value into a non-empty string, otherwise returns None.
    """
    if isinstance(value, str) and value.strip():
        return value

    return None


def _has_unsupported_terms(value: object) -> bool:
    """
    Returns whether unsupported terms are present.
    """
    return isinstance(value, list) and len(value) > 0


def resolve_core_mapping_fields(
    mapped_metric_column: object,
    previous_metric_column: object,
    default_metric_column: object,
    mapped_time_column: object,
    previous_time_column: object,
    default_time_column: object,
    aggregation_from_evidence: object,
    mapped_aggregation: object,
    previous_aggregation: object,
    unsupported_terms: object,
) -> tuple[str | None, str | None, str | None]:
    """
    Resolves core mapping fields used for SQL generation.

    Rules:
    - If unsupported terms exist, do not trust the metric column.
    - Prefer mapped values over previous state values.
    - Fall back to requirement defaults where applicable.
    - Prefer deterministic aggregation evidence over mapper aggregation.
    """
    if _has_unsupported_terms(unsupported_terms):
        metric_column = None
    else:
        metric_column = (
            _to_string_or_none(mapped_metric_column)
            or _to_string_or_none(previous_metric_column)
            or _to_string_or_none(default_metric_column)
        )

    time_column = (
        _to_string_or_none(mapped_time_column)
        or _to_string_or_none(previous_time_column)
        or _to_string_or_none(default_time_column)
    )

    aggregation = (
        _to_string_or_none(aggregation_from_evidence)
        or _to_string_or_none(mapped_aggregation)
        or _to_string_or_none(previous_aggregation)
    )

    return metric_column, time_column, aggregation

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


def _to_limit_or_none(value: object) -> int | None:
    """
    Converts a raw value into a valid limit-like integer.

    This keeps the previous truthy fallback behavior:
    - 0 is treated as missing
    - non-integer values are treated as missing
    """
    if isinstance(value, bool):
        return None

    if isinstance(value, int) and value != 0:
        return value

    return None


def _extract_order_by_column(order_by: str) -> str | None:
    """
    Extracts the first column/token from an ORDER BY expression.
    """
    order_by_parts = order_by.strip().split()

    if not order_by_parts:
        return None

    return order_by_parts[0]

def resolve_order_by_and_limit(
    mapped_order_by: object,
    previous_order_by: object,
    aggregation: object,
    group_by: object,
    time_column: object,
    mapped_limit: object,
    previous_limit: object,
    default_limit: object,
) -> tuple[str | None, int]:
    """
    Resolves ORDER BY and LIMIT for the mapped query.

    Rules:
    - For aggregation queries, use mapper order_by only.
    - If aggregation has no group_by, remove order_by.
    - If aggregation order_by targets time_column but time_column is not grouped, remove order_by.
    - For raw/non-aggregation queries, prefer mapped order_by, then previous order_by.
    - For raw/non-aggregation queries, default order_by to '<time_column> ASC' when missing.
    - Resolve limit from mapped value, previous state, default value, then 1000.
    """
    has_aggregation = _to_string_or_none(aggregation) is not None
    resolved_time_column = _to_string_or_none(time_column)
    resolved_group_by = _to_string_list(group_by)

    if has_aggregation:
        order_by = _to_string_or_none(mapped_order_by)

        if not resolved_group_by:
            order_by = None
        elif order_by is not None and resolved_time_column is not None:
            order_by_column = _extract_order_by_column(order_by)

            if (
                order_by_column == resolved_time_column
                and resolved_time_column not in resolved_group_by
            ):
                order_by = None
    else:
        order_by = (
            _to_string_or_none(mapped_order_by)
            or _to_string_or_none(previous_order_by)
        )

        if order_by is None and resolved_time_column:
            order_by = f"{resolved_time_column} ASC"

    limit = (
        _to_limit_or_none(mapped_limit)
        or _to_limit_or_none(previous_limit)
        or _to_limit_or_none(default_limit)
        or 1000
    )

    return order_by, limit