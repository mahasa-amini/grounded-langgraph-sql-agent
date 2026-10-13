from langchain_core.messages import AIMessage

from agent.state import SQLAgentState
from agent.utils.evaluation_observability import record_evaluation_event


def ask_request_relationship_node(state: SQLAgentState) -> SQLAgentState:
    ambiguous_message = (
        state.get("pending_ambiguous_message")
        or state.get("latest_user_message")
        or "that message"
    )
    question = (
        f'Do you want to apply "{ambiguous_message}" to the current request, '
        "or start a new request?"
    )

    update = {
        "messages": [AIMessage(content=question)],
        "final_answer": question,
        "pending_ambiguous_message": ambiguous_message,
        "conversation_status": "active",
    }
    record_evaluation_event(
        "lifecycle_update",
        {"transition": "preserve_for_relationship_clarification", "update": update},
    )
    return update
