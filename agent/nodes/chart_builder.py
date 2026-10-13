from typing import Any

from langchain_core.messages import AIMessage

from agent.state import SQLAgentState
from agent.utils.debug import debug_node


MAX_CHART_ROWS = 1000


def build_chart_payload_node(state: SQLAgentState) -> SQLAgentState:
    """Build a deterministic chart payload from executed query results."""
    execution_error = state.get("execution_error")
    query_result = state.get("query_result") or {}
    sql_query = state.get("sql_query")

    debug_node(
        "CHART BUILDER INPUT",
        {
            "execution_error": execution_error,
            "query_result": query_result,
            "sql_query": sql_query,
            "output_type": state.get("output_type"),
            "chart_type": state.get("chart_type"),
            "selected_columns": state.get("selected_columns", []),
            "metric_column": state.get("metric_column"),
            "time_column": state.get("time_column"),
            "aggregation": state.get("aggregation"),
            "group_by": state.get("group_by", []),
        },
    )

    if execution_error:
        final_answer = (
            "I could not build the chart because the query execution failed.\n\n"
            f"Error:\n{execution_error}"
        )

        return {
            "messages": [AIMessage(content=final_answer)],
            "final_answer": final_answer,
            "chart_payload": None,
        }

    rows = query_result.get("rows") or []
    row_count = query_result.get("row_count", len(rows))

    if row_count == 0:
        final_answer = "The query executed successfully, but there is no data to chart."

        if sql_query:
            final_answer += (
                "\n\nExecuted SQL:\n"
                f"```sql\n{sql_query.strip()}\n```"
            )

        return {
            "messages": [AIMessage(content=final_answer)],
            "final_answer": final_answer,
            "chart_payload": None,
        }

    chart_payload = _build_chart_payload(
        rows=rows[:MAX_CHART_ROWS],
        requested_chart_type=state.get("chart_type"),
        time_column=state.get("time_column"),
        metric_column=state.get("metric_column"),
        aggregation=state.get("aggregation"),
        group_by=state.get("group_by", []),
    )

    final_answer = _build_chart_answer(
        chart_payload=chart_payload,
        row_count=row_count,
        sql_query=sql_query,
    )

    debug_node(
        "CHART BUILDER OUTPUT",
        {
            "chart_payload": chart_payload,
            "final_answer": final_answer,
            "row_count": row_count,
        },
    )

    return {
        "messages": [AIMessage(content=final_answer)],
        "final_answer": final_answer,
        "chart_payload": chart_payload,
    }


def _build_chart_payload(
    rows: list[dict[str, Any]],
    requested_chart_type: str | None,
    time_column: str | None,
    metric_column: str | None,
    aggregation: str | None,
    group_by: list[str],
) -> dict[str, Any]:
    columns = _get_columns(rows)

    x_key = _resolve_x_key(
        rows=rows,
        columns=columns,
        requested_chart_type=requested_chart_type,
        time_column=time_column,
        group_by=group_by,
    )

    y_key = _resolve_y_key(
        rows=rows,
        columns=columns,
        metric_column=metric_column,
        aggregation=aggregation,
        x_key=x_key,
    )

    chart_type = _resolve_chart_type(
        requested_chart_type=requested_chart_type,
        rows=rows,
        x_key=x_key,
        y_key=y_key,
    )

    if chart_type == "table" or not x_key or not y_key:
        return {
            "type": "table",
            "title": "Query result",
            "columns": columns,
            "data": rows,
        }

    return {
        "type": chart_type,
        "title": _build_chart_title(chart_type=chart_type, y_key=y_key, x_key=x_key),
        "x_key": x_key,
        "x_label": _humanize_key(x_key),
        "y_key": y_key,
        "y_label": _humanize_key(y_key),
        "series": [
            {
                "data_key": y_key,
                "label": _humanize_key(y_key),
            }
        ],
        "data": rows,
    }


def _resolve_chart_type(
    requested_chart_type: str | None,
    rows: list[dict[str, Any]],
    x_key: str | None,
    y_key: str | None,
) -> str:
    if requested_chart_type in {"line", "bar", "table"}:
        return requested_chart_type

    if not x_key or not y_key:
        return "table"

    first_value = rows[0].get(x_key)

    if _looks_like_time_key(x_key) or _looks_like_time_value(first_value):
        return "line"

    return "bar"


def _resolve_x_key(
    rows: list[dict[str, Any]],
    columns: list[str],
    requested_chart_type: str | None,
    time_column: str | None,
    group_by: list[str],
) -> str | None:
    if not rows or not columns:
        return None

    if requested_chart_type == "line":
        time_key = _find_time_key(columns=columns, time_column=time_column)

        if time_key:
            return time_key

    for column in group_by:
        if column in columns:
            return column

    time_key = _find_time_key(columns=columns, time_column=time_column)

    if time_key:
        return time_key

    for column in columns:
        if not _is_numeric_value(rows[0].get(column)):
            return column

    return columns[0]


def _resolve_y_key(
    rows: list[dict[str, Any]],
    columns: list[str],
    metric_column: str | None,
    aggregation: str | None,
    x_key: str | None,
) -> str | None:
    if not rows or not columns:
        return None

    preferred_keys = _build_preferred_y_keys(
        metric_column=metric_column,
        aggregation=aggregation,
    )

    for key in preferred_keys:
        if key in columns and key != x_key and _is_numeric_value(rows[0].get(key)):
            return key

    for column in columns:
        if column != x_key and _is_numeric_value(rows[0].get(column)):
            return column

    return None


def _build_preferred_y_keys(
    metric_column: str | None,
    aggregation: str | None,
) -> list[str]:
    preferred_keys: list[str] = []

    if aggregation and metric_column:
        preferred_keys.append(f"{aggregation}_{metric_column}")

    if metric_column:
        preferred_keys.append(metric_column)

    preferred_keys.extend(
        [
            "avg_value",
            "sum_value",
            "min_value",
            "max_value",
            "count",
            "row_count",
            "value",
        ]
    )

    return preferred_keys


def _find_time_key(columns: list[str], time_column: str | None) -> str | None:
    if time_column and time_column in columns:
        return time_column

    for column in columns:
        if _looks_like_time_key(column):
            return column

    return None


def _get_columns(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []

    return list(rows[0].keys())


def _is_numeric_value(value: Any) -> bool:
    if isinstance(value, bool):
        return False

    return isinstance(value, (int, float))


def _looks_like_time_key(key: str | None) -> bool:
    if not key:
        return False

    normalized_key = key.lower()

    return normalized_key in {"timestamp", "time", "date", "datetime"} or normalized_key.endswith(
        "_time"
    )


def _looks_like_time_value(value: Any) -> bool:
    if not isinstance(value, str):
        return False

    return "T" in value or "+00" in value or value.count("-") >= 2


def _build_chart_title(chart_type: str, y_key: str, x_key: str) -> str:
    return f"{_humanize_key(y_key)} by {_humanize_key(x_key)} ({chart_type})"


def _humanize_key(key: str | None) -> str:
    if not key:
        return ""

    return str(key).replace("_", " ").strip().capitalize()


def _build_chart_answer(
    chart_payload: dict[str, Any],
    row_count: int,
    sql_query: str | None,
) -> str:
    chart_type = chart_payload.get("type", "chart")

    if chart_type == "table":
        message = f"The query executed successfully and returned {row_count} row(s). I prepared a table payload."
    else:
        message = (
            f"The query executed successfully and returned {row_count} row(s). "
            f"I prepared a {chart_type} chart payload."
        )

    if sql_query:
        message += (
            "\n\nExecuted SQL:\n"
            f"```sql\n{sql_query.strip()}\n```"
        )

    return message