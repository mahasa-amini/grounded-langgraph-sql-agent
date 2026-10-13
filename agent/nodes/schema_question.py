from langchain_core.messages import AIMessage

from agent.chains.schema_question_chain import get_schema_question_chain
from agent.state import SQLAgentState
from models.structured_outputs import SchemaQuestionAnswerOutput


def answer_schema_question_node(state: SQLAgentState) -> SQLAgentState:
    chain = get_schema_question_chain()

    raw_result = chain.invoke(
        {
            "latest_user_message": state.get("latest_user_message") or "",
            "public_schema_context": state.get("public_schema_context") or {},
            "request_status": state.get("request_status") or "idle",
            "active_request_type": state.get("active_request_type"),
            "missing_slots": state.get("missing_slots", []),
        }
    )

    result = SchemaQuestionAnswerOutput.model_validate(raw_result)

    return {
        "messages": [AIMessage(content=result.answer)],
        "final_answer": result.answer,
    }