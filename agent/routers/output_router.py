from typing import Literal

from agent.state import SQLAgentState
from agent.utils.debug import debug_node


ExecutionOutputRoute = Literal["text", "chart"]


def route_after_sql_execution(state: SQLAgentState) -> ExecutionOutputRoute:
    """Route executed query results to text or chart response builders."""
    output_type = state.get("output_type")

    if output_type == "chart":
        route: ExecutionOutputRoute = "chart"
    else:
        route = "text"

    debug_node(
        "ROUTE AFTER SQL EXECUTION",
        {
            "output_type": output_type,
            "chart_type": state.get("chart_type"),
            "execution_error": state.get("execution_error"),
            "route": route,
        },
    )

    return route