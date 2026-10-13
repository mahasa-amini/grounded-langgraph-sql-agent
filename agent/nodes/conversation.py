from langchain_core.messages import AIMessage

from agent.state import SQLAgentState


def answer_goodbye_node(state: SQLAgentState) -> SQLAgentState:
    message = "Goodbye. The conversation has ended."

    return {
        "messages": [AIMessage(content=message)],
        "final_answer": message,
        "conversation_status": "ended",
    }

def wait_for_next_message_node(state: SQLAgentState) -> SQLAgentState:
    return {
        "conversation_status": state.get("conversation_status") or "active",
    }