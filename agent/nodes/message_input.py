from typing import cast

from agent.state import SQLAgentState
from agent.utils.debug import debug_node


def receive_user_message_node(state: SQLAgentState) -> SQLAgentState:
    messages = state.get("messages", [])

    latest_user_message: str | None = None

    for message in reversed(messages):
        if message.type == "human":
            content = message.content

            if isinstance(content, str):
                latest_user_message = content
            else:
                latest_user_message = str(content)

            break

    update = {
        "latest_user_message": latest_user_message,
        "conversation_status": state.get("conversation_status") or "active",
        "request_status": state.get("request_status") or "idle",
        "clarification_answers": state.get("clarification_answers", []),
    }

    debug_node(
        "RECEIVE USER MESSAGE",
        {
            "latest_user_message": latest_user_message,
            "previous_request_status": state.get("request_status"),
            "next_request_status": update["request_status"],
            "previous_active_request_type": state.get("active_request_type"),
            "previous_active_request_text": state.get("active_request_text"),
            "previous_missing_slots": state.get("missing_slots", []),
        },
    )

    return cast(SQLAgentState, update)