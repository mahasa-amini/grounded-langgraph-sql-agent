from langchain_core.messages import AIMessage

from agent.chains.general_chat_chain import get_general_chat_chain
from agent.state import SQLAgentState
from models.structured_outputs import GeneralChatOutput


def answer_general_chat_node(state: SQLAgentState) -> SQLAgentState:
    chain = get_general_chat_chain()

    raw_result = chain.invoke(
        {
            "latest_user_message": state.get("latest_user_message") or "",
            "conversation_status": state.get("conversation_status") or "active",
            "request_status": state.get("request_status") or "idle",
            "active_request_type": state.get("active_request_type"),
        }
    )

    result = GeneralChatOutput.model_validate(raw_result)

    return {
        "messages": [AIMessage(content=result.answer)],
        "final_answer": result.answer,
    }