from typing import Any

from langchain_core.messages import AIMessage

from agent.state import SQLAgentState
from agent.utils.debug import debug_node


MAX_PREVIEW_ROWS = 10


def build_text_response_node(state: SQLAgentState) -> SQLAgentState:
    """Build a deterministic text response from executed query results."""
    execution_error = state.get("execution_error")
    query_result = state.get("query_result") or {}
    sql_query = state.get("sql_query")

    debug_node(
        "TEXT RESPONSE BUILDER INPUT",
        {
            "execution_error": execution_error,
            "query_result": query_result,
            "sql_query": sql_query,
            "output_type": state.get("output_type"),
            "chart_type": state.get("chart_type"),
        },
    )

    if execution_error:
        final_answer = (
            "I could not execute the query successfully.\n\n"
            f"Error:\n{execution_error}"
        )

        return {
            "messages": [AIMessage(content=final_answer)],
            "final_answer": final_answer,
        }

    rows = query_result.get("rows") or []
    row_count = query_result.get("row_count", len(rows))

    if row_count == 0:
        final_answer = _build_no_rows_answer(sql_query=sql_query)

        return {
            "messages": [AIMessage(content=final_answer)],
            "final_answer": final_answer,
        }

    final_answer = _build_rows_answer(
        rows=rows,
        row_count=row_count,
        sql_query=sql_query,
    )

    debug_node(
        "TEXT RESPONSE BUILDER OUTPUT",
        {
            "final_answer": final_answer,
            "row_count": row_count,
        },
    )

    return {
        "messages": [AIMessage(content=final_answer)],
        "final_answer": final_answer,
    }


def _build_no_rows_answer(sql_query: str | None) -> str:
    parts = [
        "The query executed successfully, but no rows matched your request.",
    ]

    if sql_query:
        parts.extend(
            [
                "",
                "Executed SQL:",
                f"```sql\n{sql_query.strip()}\n```",
            ]
        )

    return "\n".join(parts)


def _build_rows_answer(
    rows: list[dict[str, Any]],
    row_count: int,
    sql_query: str | None,
) -> str:
    preview_rows = rows[:MAX_PREVIEW_ROWS]

    parts = [
        f"The query executed successfully and returned {row_count} row(s).",
        "",
        _format_rows_as_markdown_table(preview_rows),
    ]

    if row_count > MAX_PREVIEW_ROWS:
        parts.extend(
            [
                "",
                f"Showing the first {MAX_PREVIEW_ROWS} row(s).",
            ]
        )

    if sql_query:
        parts.extend(
            [
                "",
                "Executed SQL:",
                f"```sql\n{sql_query.strip()}\n```",
            ]
        )

    return "\n".join(parts)


def _format_rows_as_markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""

    columns = list(rows[0].keys())

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"

    body_lines = []

    for row in rows:
        values = [
            _format_cell_value(row.get(column))
            for column in columns
        ]
        body_lines.append("| " + " | ".join(values) + " |")

    return "\n".join([header, separator, *body_lines])


def _format_cell_value(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, float):
        return f"{value:.6g}"

    text = str(value)
    return text.replace("\n", " ").replace("|", "\\|")