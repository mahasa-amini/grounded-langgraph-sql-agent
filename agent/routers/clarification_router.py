from typing import Literal

from agent.state import SQLAgentState
from app.config import settings


def route_clarification_limit(
    state: SQLAgentState,
) -> Literal["can_continue", "limit_exceeded"]:
    clarification_count = state.get("clarification_count", 0)
    max_count = state.get("max_clarification_count", settings.max_clarification_count)

    if clarification_count >= max_count:
        return "limit_exceeded"

    return "can_continue"