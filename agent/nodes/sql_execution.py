from typing import Any

from agent.state import SQLAgentState
from agent.utils.debug import debug_node
from db.bigquery_client import execute_bigquery_select


def execute_sql_node(state: SQLAgentState) -> SQLAgentState:
    """Execute the generated SQL query and store the normalized result in state."""
    sql_query = state.get("sql_query")

    debug_node(
        "SQL EXECUTION INPUT",
        {
            "sql_query": sql_query,
            "sql_is_safe": state.get("sql_is_safe"),
            "output_type": state.get("output_type"),
            "chart_type": state.get("chart_type"),
        },
    )

    if not sql_query:
        execution_error = "No SQL query is available for execution."

        debug_node(
            "SQL EXECUTION SKIPPED",
            {
                "execution_error": execution_error,
            },
        )

        return {
            "query_result": None,
            "execution_error": execution_error,
        }

    execution_result: dict[str, Any] = execute_bigquery_select(sql_query)
    executable_sql = execution_result.get("sql_query") or sql_query

    if not execution_result.get("success"):
        execution_error = execution_result.get("error") or "Unknown SQL execution error."

        debug_node(
            "SQL EXECUTION FAILED",
            {
                "executable_sql": executable_sql,
                "execution_error": execution_error,
                "row_count": execution_result.get("row_count"),
                "job_id": execution_result.get("job_id"),
                "total_bytes_processed": execution_result.get("total_bytes_processed"),
            },
        )

        return {
            "sql_query": executable_sql,
            "query_result": execution_result,
            "execution_error": execution_error,
        }

    debug_node(
        "SQL EXECUTION SUCCESS",
        {
            "executable_sql": executable_sql,
            "row_count": execution_result.get("row_count"),
            "job_id": execution_result.get("job_id"),
            "total_bytes_processed": execution_result.get("total_bytes_processed"),
            "rows": execution_result.get("rows"),
        },
    )

    return {
        "sql_query": executable_sql,
        "query_result": execution_result,
        "execution_error": None,
    }