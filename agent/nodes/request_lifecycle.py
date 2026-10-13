from typing import cast

from langchain_core.messages import AIMessage

from agent.state import SQLAgentState
from agent.utils.debug import debug_node
from agent.utils.confirmation import (
    resolve_confirmation_answer_from_suggestion,
    try_apply_affirmative_location_confirmation,
)
from agent.utils.evaluation_observability import record_evaluation_event


def reset_active_request_node(state: SQLAgentState) -> SQLAgentState:
    message = (
        "No problem. I cleared the active request. "
        "You can start a new data or chart request whenever you are ready."
    )

    update = {
        "messages": [AIMessage(content=message)],
        "final_answer": message,
        "request_status": "idle",
        "active_request_type": None,
        "active_request_text": None,
        "intent": None,
        "output_type": None,
        "chart_type": None,
        "table_name": None,
        "unsupported_terms": [],
        "raw_metric": None,
        "ambiguous_terms": [],
        "clarification_question": None,
        "clarification_answers": [],
        "clarification_count": 0,
        "mapping_input_text": None,
        "active_request_action": "none",
        "classification_valid": True,
        "classification_error": None,
        "model_classification": None,
        "classification_safeguard": None,
        "request_transition_valid": True,
        "correction_targets": [],
        "classified_request_type": None,
        "pending_ambiguous_message": None,
        "pending_ambiguous_request_type": None,
        "pending_ambiguous_correction_targets": [],
        "request_evidence": None,
        "parsed_time_range": None,
        "required_slots": [],
        "missing_slots": [],
        "is_complete": False,
        "selected_columns": [],
        "metric_column": None,
        "time_column": None,
        "time_range": None,
        "filters": [],
        "aggregation": None,
        "group_by": [],
        "order_by": None,
        "having_conditions": [],
        "distinct": False,
        "limit": None,
        "sql_query": None,
        "sql_explanation": None,
        "sql_generation_provenance": None,
        "sql_compilation_errors": [],
        "sql_is_safe": False,
        "sql_validation_errors": [],
        "query_result": None,
        "execution_error": None,
        "chart_payload": None,
        "error": None,
        "evidence_suggestions": {},
    }
    record_evaluation_event("lifecycle_update", {"transition": "reset", "update": update})
    return cast(SQLAgentState, update)


def _fresh_request_update(
    request_text: str,
    active_request_type: str | None,
) -> dict:
    output_type = "chart" if active_request_type == "chart" else "text"
    return {
        "request_status": "collecting",
        "active_request_type": active_request_type or "data",
        "intent": None,
        "output_type": output_type,
        "chart_type": None,
        "table_name": None,
        "active_request_text": request_text,
        "clarification_question": None,
        "clarification_answers": [],
        "clarification_count": 0,
        "mapping_input_text": request_text,
        "active_request_action": "none",
        "classification_valid": True,
        "classification_error": None,
        "request_transition_valid": True,
        "correction_targets": [],
        "classified_request_type": None,
        "pending_ambiguous_message": None,
        "pending_ambiguous_request_type": None,
        "pending_ambiguous_correction_targets": [],
        "request_evidence": None,
        "parsed_time_range": None,
        "evidence_suggestions": {},
        "required_slots": [],
        "missing_slots": [],
        "is_complete": False,
        "unsupported_terms": [],
        "ambiguous_terms": [],
        "raw_metric": None,
        "selected_columns": [],
        "metric_column": None,
        "time_column": None,
        "time_range": None,
        "filters": [],
        "having_conditions": [],
        "distinct": False,
        "aggregation": None,
        "group_by": [],
        "order_by": None,
        "limit": None,
        "sql_query": None,
        "sql_explanation": None,
        "sql_generation_provenance": None,
        "sql_compilation_errors": [],
        "sql_is_safe": False,
        "sql_validation_errors": [],
        "query_result": None,
        "execution_error": None,
        "final_answer": None,
        "chart_payload": None,
        "error": None,
    }


def _clear_corrected_fields(update: dict, targets: set[str]) -> None:
    if "metric" in targets:
        update.update(
            raw_metric=None,
            metric_column=None,
            selected_columns=[],
            filters=[
                item
                for item in update.get("filters", [])
                if item.get("column") != "type"
            ],
        )
    if "location" in targets:
        update["filters"] = [
            item
            for item in update.get("filters", [])
            if item.get("column") not in {"city", "district"}
        ]
    if "filters" in targets:
        update.update(filters=[], having_conditions=[])
    if "time_range" in targets:
        update.update(time_range=None, parsed_time_range=None)
    if "aggregation" in targets:
        update.update(aggregation=None, having_conditions=[], order_by=None)
    if "chart_type" in targets:
        update["chart_type"] = None
    if "output_type" in targets:
        update["chart_type"] = None
    if "group_by" in targets:
        update.update(group_by=[], selected_columns=[])
    if "order_by" in targets:
        update["order_by"] = None
    if "limit" in targets:
        update["limit"] = None


def start_or_update_active_request_node(state: SQLAgentState) -> SQLAgentState:

    latest_user_message = state.get("latest_user_message") or ""
    user_message_type = state.get("user_message_type")
    request_status = state.get("request_status") or "idle"
    action = state.get("active_request_action") or "none"

    allowed_actions = {
        "new_request", "clarification_answer", "correction",
        "continue_ambiguous", "new_request_from_ambiguous",
    }
    if state.get("classification_valid") is not True or action not in allowed_actions:
        update = cast(SQLAgentState, {
            "request_transition_valid": False,
            "classification_valid": False,
            "classification_error": state.get("classification_error")
            or f"Lifecycle rejected action {action!r}.",
        })
        record_evaluation_event("lifecycle_update", {"update": dict(update)})
        return update

    if user_message_type == "chart_request":
        active_request_type = "chart"
        output_type = "chart"
    elif user_message_type == "data_request":
        active_request_type = "data"
        output_type = "text"
    else:
        active_request_type = state.get("active_request_type")
        output_type = state.get("output_type")

    classified_request_type = state.get("classified_request_type")
    if action in {"clarification_answer", "correction"} and classified_request_type:
        active_request_type = classified_request_type
        output_type = "chart" if classified_request_type == "chart" else "text"

    starts_fresh = action in {"new_request", "new_request_from_ambiguous"}

    if starts_fresh:
        if action == "new_request_from_ambiguous":
            request_text = state.get("pending_ambiguous_message") or latest_user_message
            active_request_type = state.get("pending_ambiguous_request_type")
            if active_request_type not in {"data", "chart"}:
                update = cast(SQLAgentState, {
                    "request_transition_valid": False,
                    "classification_valid": False,
                    "classification_error": "Fresh ambiguous request has no validated request type.",
                })
                record_evaluation_event("lifecycle_update", {"update": dict(update)})
                return update
        else:
            request_text = latest_user_message

        if active_request_type not in {"data", "chart"}:
            update = cast(SQLAgentState, {
                "request_transition_valid": False,
                "classification_valid": False,
                "classification_error": "Fresh request has no validated request type.",
            })
            record_evaluation_event("lifecycle_update", {"update": dict(update)})
            return update

        update = _fresh_request_update(request_text, active_request_type)

        debug_node(
            "START ACTIVE REQUEST",
            {
                "latest_user_message": latest_user_message,
                "user_message_type": user_message_type,
                "previous_request_status": request_status,
                "active_request_type": active_request_type,
                "output_type": output_type,
                "active_request_text": request_text,
                "mapping_input_text": update["mapping_input_text"],
                "missing_slots": update["missing_slots"],
                "evidence_suggestions": update["evidence_suggestions"],
            },
        )

        record_evaluation_event("lifecycle_update", {"update": update})
        return cast(SQLAgentState, update)

    previous_answers = state.get("clarification_answers", []) or []
    previous_missing_slots = state.get("missing_slots", []) or []
    previous_evidence_suggestions = state.get("evidence_suggestions") or {}
    previous_active_request_text = state.get("active_request_text") or ""

    if action == "continue_ambiguous":
        message_to_apply = state.get("pending_ambiguous_message") or latest_user_message
    else:
        message_to_apply = latest_user_message

    location_confirmation_result = try_apply_affirmative_location_confirmation(
        latest_user_message=message_to_apply,
        active_request_text=previous_active_request_text,
        missing_slots=previous_missing_slots,
        evidence_suggestions=previous_evidence_suggestions,
    )

    location_confirmation_applied = bool(
        location_confirmation_result.get("applied")
    )

    if location_confirmation_applied:
        active_request_text = location_confirmation_result["active_request_text"]
        resolved_clarification_answer = location_confirmation_result[
            "resolved_clarification_answer"
        ]
        missing_slots = location_confirmation_result["missing_slots"]
        evidence_suggestions = location_confirmation_result["evidence_suggestions"]
    else:
        active_request_text = previous_active_request_text
        resolved_clarification_answer = resolve_confirmation_answer_from_suggestion(
            latest_user_message=message_to_apply,
            evidence_suggestions=previous_evidence_suggestions,
        )
        missing_slots = previous_missing_slots
        evidence_suggestions = previous_evidence_suggestions

    updated_answers = [
        *previous_answers,
        resolved_clarification_answer,
    ]

    mapping_input_parts = []

    if active_request_text:
        mapping_input_parts.append(str(active_request_text))

    for answer in updated_answers:
        if answer:
            mapping_input_parts.append(str(answer))

    mapping_input_text = "\n".join(mapping_input_parts)

    update = {
        "request_status": "collecting",
        "active_request_type": active_request_type,
        "output_type": output_type,
        "active_request_text": active_request_text,
        "clarification_answers": updated_answers,
        "mapping_input_text": mapping_input_text,
        "request_evidence": None,
        "parsed_time_range": None,
        "evidence_suggestions": evidence_suggestions,
        "missing_slots": missing_slots,
        "final_answer": None,
        "sql_query": None,
        "sql_explanation": None,
        "sql_generation_provenance": None,
        "sql_compilation_errors": [],
        "sql_is_safe": False,
        "sql_validation_errors": [],
        "query_result": None,
        "execution_error": None,
        "chart_payload": None,
        "is_complete": False,
        "active_request_action": "none",
        "classification_valid": True,
        "classification_error": None,
        "request_transition_valid": True,
        "pending_ambiguous_message": None,
        "pending_ambiguous_request_type": None,
        "pending_ambiguous_correction_targets": [],
        "error": None,
    }

    correction_targets = set(state.get("correction_targets", []) or [])
    if action in {"correction", "continue_ambiguous"} and correction_targets:
        update.update(
            filters=list(state.get("filters", []) or []),
            having_conditions=list(state.get("having_conditions", []) or []),
        )
        _clear_corrected_fields(update, correction_targets)
        update["correction_targets"] = sorted(correction_targets)
    else:
        update["correction_targets"] = []

    if "output_type" in correction_targets:
        corrected_type = state.get("classified_request_type")
        if corrected_type in {"data", "chart"}:
            update["active_request_type"] = corrected_type
            update["output_type"] = "chart" if corrected_type == "chart" else "text"

    debug_node(
        "UPDATE ACTIVE REQUEST",
        {
            "latest_user_message": latest_user_message,
            "resolved_clarification_answer": resolved_clarification_answer,
            "user_message_type": user_message_type,
            "previous_request_status": request_status,
            "active_request_type": active_request_type,
            "output_type": output_type,
            "previous_active_request_text": previous_active_request_text,
            "active_request_text": active_request_text,
            "previous_clarification_answers": previous_answers,
            "updated_clarification_answers": updated_answers,
            "previous_missing_slots": previous_missing_slots,
            "missing_slots": missing_slots,
            "previous_evidence_suggestions": previous_evidence_suggestions,
            "evidence_suggestions": evidence_suggestions,
            "location_confirmation_applied": location_confirmation_applied,
            "mapping_input_text": mapping_input_text,
        },
    )

    record_evaluation_event("lifecycle_update", {"update": update})
    return cast(SQLAgentState, update)


def mark_request_done_node(state: SQLAgentState) -> SQLAgentState:
    final_answer = state.get("final_answer")
    sql_query = state.get("sql_query")

    update = {
        # Mark the request as finished.
        "request_status": "idle",
        "active_request_type": None,
        "active_request_text": None,
        "mapping_input_text": None,

        # Clear clarification state.
        "clarification_question": None,
        "clarification_answers": [],
        "clarification_count": 0,
        "active_request_action": "none",
        "classification_valid": True,
        "classification_error": None,
        "request_transition_valid": True,
        "correction_targets": [],
        "classified_request_type": None,
        "pending_ambiguous_message": None,
        "pending_ambiguous_request_type": None,
        "pending_ambiguous_correction_targets": [],

        # Clear evidence and parsing state.
        "request_evidence": None,
        "parsed_time_range": None,
        "evidence_suggestions": {},

        # Clear mapping state.
        "required_slots": [],
        "missing_slots": [],
        "is_complete": False,
        "unsupported_terms": [],
        "ambiguous_terms": [],
        "raw_metric": None,
        "selected_columns": [],
        "metric_column": None,
        "time_column": None,
        "time_range": None,
        "filters": [],
        "aggregation": None,
        "group_by": [],
        "order_by": None,
        "having_conditions": [],
        "distinct": False,
        "limit": None,

        # Keep SQL/execution/chart fields for the just-finished request.
        # They will be cleared when a new request starts.
        "error": None,
    }

    if not final_answer:
        fallback_message = "The request has been completed."

        if sql_query:
            fallback_message = (
                "The request has been completed.\n\n"
                "Executed SQL:\n"
                f"```sql\n{sql_query.strip()}\n```"
            )

        update["messages"] = [AIMessage(content=fallback_message)]
        update["final_answer"] = fallback_message

    debug_node(
        "MARK REQUEST DONE",
        {
            "had_final_answer": bool(final_answer),
            "kept_sql_query": sql_query,
            "kept_query_result": state.get("query_result"),
            "execution_error": state.get("execution_error"),
            "chart_payload": state.get("chart_payload"),
            "request_status": update["request_status"],
        },
    )

    return cast(SQLAgentState, update)
