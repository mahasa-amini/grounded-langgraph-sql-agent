from agent.state import SQLAgentState
from agent.nodes.sql_safety import is_early_sql_safety_error
from agent.utils.evaluation_observability import record_evaluation_event


def route_after_early_sql_safety_check(state: SQLAgentState) -> str:
    """
    Route unsafe requests to security rejection.

    Safe requests continue to the normal message-context router.
    """

    if is_early_sql_safety_error(state.get("error")):
        return "security_reject"

    return route_message_with_context(state)


def route_message_with_context(state: SQLAgentState) -> str:
    message_type = state.get("user_message_type") or "unclear"
    request_status = state.get("request_status") or "idle"
    action = state.get("active_request_action") or "none"

    if state.get("classification_valid") is False or action == "invalid":
        route = "ambiguous_request_relationship" if request_status == "collecting" else "unclear"
        record_evaluation_event("chosen_route", {"route": route, "reason": "invalid classification"})
        return route

    if action == "cancel":
        route = "reset_request"
        record_evaluation_event("chosen_route", {"route": route})
        return route

    if action == "side_question":
        route = "schema_question" if message_type == "schema_question" else "general_chat"
        record_evaluation_event("chosen_route", {"route": route})
        return route

    if action == "ambiguous":
        record_evaluation_event("chosen_route", {"route": "ambiguous_request_relationship"})
        return "ambiguous_request_relationship"

    if action in {"continue_ambiguous", "new_request_from_ambiguous"}:
        record_evaluation_event("chosen_route", {"route": "clarification_or_update"})
        return "clarification_or_update"

    if action in {"clarification_answer", "correction"}:
        route = "clarification_or_update" if request_status == "collecting" else "unclear"
        record_evaluation_event("chosen_route", {"route": route})
        return route

    if action == "new_request":
        record_evaluation_event("chosen_route", {"route": "data_or_chart_request"})
        return "data_or_chart_request"

    if message_type == "goodbye":
        route = "goodbye"
        record_evaluation_event("chosen_route", {"route": route})
        return route

    if message_type == "general_chat":
        route = "general_chat"
        record_evaluation_event("chosen_route", {"route": route})
        return route

    if message_type == "schema_question":
        route = "schema_question"
        record_evaluation_event("chosen_route", {"route": route})
        return route

    if message_type in {"data_request", "chart_request"}:
        route = "data_or_chart_request"
        record_evaluation_event("chosen_route", {"route": route})
        return route

    if message_type == "reset_request":
        route = "reset_request"
        record_evaluation_event("chosen_route", {"route": route})
        return route

    if message_type == "clarification_or_update":
        if request_status == "collecting":
            route = "clarification_or_update"
            record_evaluation_event("chosen_route", {"route": route})
            return route

        route = "unclear"
        record_evaluation_event("chosen_route", {"route": route})
        return route

    route = "unclear"
    record_evaluation_event("chosen_route", {"route": route})
    return route
