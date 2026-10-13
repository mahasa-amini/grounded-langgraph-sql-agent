from typing import Literal

from agent.state import SQLAgentState


def route_completeness(
    state: SQLAgentState,
) -> Literal["complete", "incomplete"]:
    if state.get("is_complete"):
        return "complete"

    return "incomplete"