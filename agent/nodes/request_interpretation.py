from agent.state import SQLAgentState


def build_mapping_input_text(state: SQLAgentState) -> str:
    active_request_text = state.get("active_request_text") or ""
    clarification_answers = state.get("clarification_answers", [])

    text_parts = []

    if active_request_text:
        text_parts.append(active_request_text)

    for answer in clarification_answers:
        if answer:
            text_parts.append(answer)

    if text_parts:
        return "\n".join(text_parts)

    return state.get("latest_user_message") or ""


def interpret_user_request_node(state: SQLAgentState) -> SQLAgentState:
    user_message = build_mapping_input_text(state)
    normalized_message = user_message.lower()

    if state.get("active_request_type") == "chart":
        output_type = "chart"
    elif state.get("active_request_type") == "data":
        output_type = "text"
    else:
        output_type = (
            "chart"
            if any(term in normalized_message for term in ["chart", "graph", "plot"])
            else "text"
        )

    return {
        "intent": "retrieve_data",
        "output_type": output_type,
    }