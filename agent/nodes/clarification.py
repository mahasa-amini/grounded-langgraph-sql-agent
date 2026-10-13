from typing import cast

from langchain_core.messages import AIMessage

from agent.state import SQLAgentState
from agent.utils.debug import debug_node
from app.config import settings
from tools.clarification_question_tool import build_clarification_question_tool


def check_clarification_limit_node(state: SQLAgentState) -> SQLAgentState:
    current_count = state.get("clarification_count", 0)
    max_count = state.get("max_clarification_count", settings.max_clarification_count)

    return {
        "clarification_count": current_count,
        "max_clarification_count": max_count,
    }


def answer_clarification_node(state: SQLAgentState) -> SQLAgentState:
    missing_slots = state.get("missing_slots", [])
    current_count = state.get("clarification_count", 0)

    missing_slots = state.get("missing_slots", []) or []
    evidence_suggestions = state.get("evidence_suggestions", {}) or {}

    if "location_confirmation" in missing_slots:
        location_suggestion = evidence_suggestions.get("location", {}) or {}
        suggestion_question = location_suggestion.get("question")

        if suggestion_question:
            debug_node(
                "CLARIFICATION ANSWER",
                {
                    "missing_slots": missing_slots,
                    "evidence_suggestions": evidence_suggestions,
                    "final_answer": suggestion_question,
                },
            )

            return {
                "messages": [AIMessage(content=suggestion_question)],
                "final_answer": suggestion_question,
                "clarification_question": suggestion_question,
                "clarification_count": state.get("clarification_count", 0) + 1,
            }

    question = build_clarification_question_tool(
        missing_slots=missing_slots,
        state_snapshot=dict(state),
    )

    update = {
        "messages": [AIMessage(content=question)],
        "clarification_question": question,
        "clarification_count": current_count + 1,
        "request_status": "collecting",
        "final_answer": question,
    }

    debug_node(
        "CLARIFICATION ANSWER",
        {
            "missing_slots": missing_slots,
            "error": state.get("error"),
            "raw_metric": state.get("raw_metric"),
            "metric_column": state.get("metric_column"),
            "time_range": state.get("time_range"),
            "output_type": state.get("output_type"),
            "chart_type": state.get("chart_type"),
            "clarification_question": question,
            "clarification_question_length": len(question),
            "final_answer": update["final_answer"],
            "clarification_count_before": current_count,
            "clarification_count_after": update["clarification_count"],
        },
    )

    return cast(SQLAgentState, update)


def fail_gracefully_node(state: SQLAgentState) -> SQLAgentState:
    missing_slots = state.get("missing_slots", [])
    missing_text = ", ".join(missing_slots) if missing_slots else "unknown fields"

    message = (
        "I still do not have enough information to execute your request accurately. "
        f"The unresolved fields are: {missing_text}. "
        "Please try again with a more detailed request, including the metric, time range, "
        "location filters, output type, and any other relevant constraints."
    )

    return {
        "messages": [AIMessage(content=message)],
        "final_answer": message,
        "request_status": "idle",
    }