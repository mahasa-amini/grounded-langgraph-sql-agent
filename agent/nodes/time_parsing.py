from typing import cast

from agent.state import SQLAgentState
from agent.utils.debug import debug_node
from agent.utils.evaluation_observability import record_evaluation_event
from tools.normalize_time_range_tool import normalize_time_range_tool


def parse_time_phrase_node(state: SQLAgentState) -> SQLAgentState:
    request_evidence = state.get("request_evidence") or {}
    time_phrase = request_evidence.get("time_phrase")

    parsed_time_range = normalize_time_range_tool(time_phrase)
    record_evaluation_event(
        "time_parser",
        {"input": time_phrase, "result": parsed_time_range},
    )

    debug_node(
        "PARSE TIME PHRASE",
        {
            "time_phrase": time_phrase,
            "parsed_time_range": parsed_time_range,
            "previous_time_range": state.get("time_range"),
        },
    )

    return cast(
        SQLAgentState,
        {
            "parsed_time_range": parsed_time_range,
        },
    )
