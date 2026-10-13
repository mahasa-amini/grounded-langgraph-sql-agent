from typing import Any


def enforce_raw_query_shape(
    mapping: dict[str, Any],
    available_columns: set[str] | None = None,
) -> dict[str, Any]:
    """
    Enforce a deterministic shape for raw data queries.

    Raw query means:
    - no aggregation
    - no group_by
    - selected_columns should be [time_column, metric_column]
    """

    updated_mapping = dict(mapping)

    aggregation = updated_mapping.get("aggregation")

    if aggregation:
        return updated_mapping

    time_column = updated_mapping.get("time_column")
    metric_column = updated_mapping.get("metric_column")

    selected_columns: list[str] = []

    if time_column and (
        available_columns is None or time_column in available_columns
    ):
        selected_columns.append(str(time_column))

    if metric_column and metric_column != time_column and (
        available_columns is None or metric_column in available_columns
    ):
        selected_columns.append(str(metric_column))

    updated_mapping["group_by"] = []

    if selected_columns:
        updated_mapping["selected_columns"] = selected_columns

    if time_column:
        updated_mapping["order_by"] = f"{time_column} ASC"

    if not updated_mapping.get("limit"):
        updated_mapping["limit"] = 1000

    return updated_mapping