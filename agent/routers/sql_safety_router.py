from typing import Literal

from agent.state import SQLAgentState


def route_sql_safety(state: SQLAgentState) -> Literal["safe", "unsafe"]:
    if state.get("sql_is_safe"):
        return "safe"

    return "unsafe"