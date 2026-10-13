from typing import Any


def build_pre_guardrail_mapping(
    table_name: object,
    active_request_type: object,
    output_type: object,
    chart_type: object,
    raw_metric: object,
    selected_columns: object,
    metric_column: object,
    time_column: object,
    time_range: object,
    filters: object,
    aggregation: object,
    group_by: object,
    order_by: object,
    limit: object,
    unsupported_terms: object,
    ambiguous_terms: object,
) -> dict[str, Any]:
    """
    Builds the normalized mapping payload before request evidence guardrails.

    This function does not change values. It only centralizes the payload shape
    used by the guardrail step.
    """
    return {
        "table_name": table_name,
        "active_request_type": active_request_type,
        "output_type": output_type,
        "chart_type": chart_type,
        "raw_metric": raw_metric,
        "selected_columns": selected_columns,
        "metric_column": metric_column,
        "time_column": time_column,
        "time_range": time_range,
        "filters": filters,
        "aggregation": aggregation,
        "group_by": group_by,
        "order_by": order_by,
        "limit": limit,
        "unsupported_terms": unsupported_terms,
        "ambiguous_terms": ambiguous_terms,
    }