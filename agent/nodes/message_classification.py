from typing import cast

from agent.chains.message_classification_chain import get_message_classification_chain
from agent.state import SQLAgentState
from agent.utils.debug import debug_node
from agent.utils.evaluation_observability import (
    compact_model_value,
    is_evaluation_observability_enabled,
    record_evaluation_event,
)
from models.structured_outputs import MessageClassificationOutput
from tools.normalize_time_range_tool import normalize_bare_date_tool
from tools.normalize_location_tool import normalize_exact_location_tool


_CORRECTION_TARGETS = {
    "metric", "location", "time_range", "aggregation", "chart_type",
    "output_type", "group_by", "order_by", "limit", "filters",
}
_EXACT_CANCELLATIONS = {"forget that", "cancel this request"}
_EXACT_CONTINUE_AMBIGUITY = {
    "apply it to the current request",
    "apply this to the current request",
    "apply that to the current request",
    "continue the current request",
    "change the current request",
}
_EXACT_NEW_REQUEST_AMBIGUITY = {
    "start a new request",
    "make it a new request",
}


def _compact_collected_fields(state: SQLAgentState) -> dict:
    return {
        "raw_metric": state.get("raw_metric"),
        "metric_column": state.get("metric_column"),
        "filters": state.get("filters", []),
        "time_range": state.get("time_range"),
        "aggregation": state.get("aggregation"),
        "group_by": state.get("group_by", []),
        "order_by": state.get("order_by"),
        "limit": state.get("limit"),
        "output_type": state.get("output_type"),
        "chart_type": state.get("chart_type"),
    }


def _compact_recent_turns(state: SQLAgentState) -> list[dict[str, str]]:
    turns = []
    for message in (state.get("messages") or [])[-4:]:
        content = getattr(message, "content", "")
        if not isinstance(content, str):
            content = str(content)
        turns.append({"role": getattr(message, "type", "message"), "content": content[:500]})
    return turns


def _is_exact_cancellation(text: str) -> bool:
    # Deliberately narrow: quotes, prefixes, and questions do not match.
    normalized = " ".join(text.strip().lower().split()).rstrip(".! ")
    return normalized in _EXACT_CANCELLATIONS


def _exact_ambiguity_resolution(text: str) -> str | None:
    # Deliberately whole-message only. Quoted choices and longer prose do not match.
    normalized = " ".join(text.strip().lower().split()).rstrip(".! ")
    if normalized in _EXACT_CONTINUE_AMBIGUITY:
        return "continue_ambiguous"
    if normalized in _EXACT_NEW_REQUEST_AMBIGUITY:
        return "new_request_from_ambiguous"
    return None


def _is_pending_bare_date_answer(state: SQLAgentState, latest: str) -> bool:
    return bool(
        (state.get("request_status") or "idle") == "collecting"
        and state.get("active_request_text")
        and not state.get("pending_ambiguous_message")
        and "time_range" in (state.get("missing_slots") or [])
        and normalize_bare_date_tool(latest) is not None
    )


def _canonicalize_pending_bare_date(
    result: MessageClassificationOutput,
    state: SQLAgentState,
) -> MessageClassificationOutput:
    return result.model_copy(
        update={
            "message_type": "clarification_or_update",
            "active_request_action": "clarification_answer",
            "correction_targets": [],
            "resolved_request_type": state.get("active_request_type"),
        }
    )


def _is_pending_exact_location_answer(state: SQLAgentState, latest: str) -> bool:
    return bool(
        (state.get("request_status") or "idle") == "collecting"
        and state.get("active_request_text")
        and not state.get("pending_ambiguous_message")
        and "location" in (state.get("missing_slots") or [])
        and normalize_exact_location_tool(
            latest,
            state.get("semantic_context") or {},
        )
    )


def _canonicalize_pending_location(
    result: MessageClassificationOutput,
    state: SQLAgentState,
) -> MessageClassificationOutput:
    return result.model_copy(
        update={
            "message_type": "clarification_or_update",
            "active_request_action": "clarification_answer",
            "correction_targets": [],
            "resolved_request_type": state.get("active_request_type"),
        }
    )


def _is_idle_request_initialization(
    result: MessageClassificationOutput,
    state: SQLAgentState,
) -> bool:
    return bool(
        (state.get("request_status") or "idle") == "idle"
        and not state.get("active_request_text")
        and not state.get("pending_ambiguous_message")
        and result.message_type in {"data_request", "chart_request"}
        and result.active_request_action == "none"
    )


def _canonicalize_idle_request(
    result: MessageClassificationOutput,
) -> MessageClassificationOutput:
    request_type = "chart" if result.message_type == "chart_request" else "data"
    return result.model_copy(
        update={
            "active_request_action": "new_request",
            "correction_targets": [],
            "resolved_request_type": request_type,
        }
    )


def _classification_error(result: MessageClassificationOutput, state: SQLAgentState) -> str | None:
    action = result.active_request_action
    message_type = result.message_type
    collecting = (state.get("request_status") or "idle") == "collecting"
    pending = bool(state.get("pending_ambiguous_message"))
    targets = set(result.correction_targets)

    if not targets.issubset(_CORRECTION_TARGETS):
        return "unsupported correction target"
    if pending and action in {"clarification_answer", "correction", "ambiguous"}:
        return "saved ambiguity must be resolved before another request mutation"
    if action == "none":
        if message_type in {"data_request", "chart_request", "reset_request", "clarification_or_update"}:
            return "none is incompatible with a request transition"
        return None
    if action == "cancel":
        return None if message_type == "reset_request" else "cancel requires reset_request"
    if action == "side_question":
        return None if collecting and message_type == "schema_question" else "side_question requires an active schema question"
    if action == "new_request":
        if message_type not in {"data_request", "chart_request"}:
            return "new_request requires a data or chart request"
        expected_type = "chart" if message_type == "chart_request" else "data"
        if result.resolved_request_type not in {None, expected_type}:
            return "resolved request type conflicts with message type"
        return None
    if action == "clarification_answer":
        return None if collecting and message_type == "clarification_or_update" else "clarification_answer requires an active request update"
    if action == "correction":
        if not collecting or message_type != "clarification_or_update":
            return "correction requires an active request update"
        return None if targets else "correction requires at least one target"
    if action == "ambiguous":
        return None if collecting and not pending else "ambiguous requires an active request without a saved ambiguity"
    if action == "continue_ambiguous":
        if not collecting or not pending:
            return "continue_ambiguous requires a saved ambiguous message"
        saved_targets = set(state.get("pending_ambiguous_correction_targets", []) or [])
        if targets and targets != saved_targets:
            return "resolution targets conflict with the saved ambiguous message"
        return None
    if action == "new_request_from_ambiguous":
        return None if collecting and pending else "new_request_from_ambiguous requires a saved ambiguous message"
    return "unknown active request action"


def _invalid_update(
    state: SQLAgentState,
    error: str,
    model_classification: dict | None = None,
) -> SQLAgentState:
    latest = state.get("latest_user_message") or ""
    pending = state.get("pending_ambiguous_message")
    pending_request_type = state.get("pending_ambiguous_request_type")
    if pending_request_type not in {"data", "chart"} and model_classification:
        message_type = model_classification.get("message_type")
        resolved_type = model_classification.get("resolved_request_type")
        coherent_type = "chart" if message_type == "chart_request" else "data"
        if (
            message_type in {"data_request", "chart_request"}
            and resolved_type in {None, coherent_type}
        ):
            pending_request_type = coherent_type
    return cast(SQLAgentState, {
        "user_message_type": "unclear",
        "classification_reason": "Classification could not safely determine the request relationship.",
        "active_request_action": "invalid",
        "classification_valid": False,
        "classification_error": error,
        "model_classification": model_classification,
        "classification_safeguard": None,
        "correction_targets": [],
        "classified_request_type": None,
        "pending_ambiguous_message": pending or latest or None,
        "pending_ambiguous_request_type": pending_request_type,
        "pending_ambiguous_correction_targets": list(
            state.get("pending_ambiguous_correction_targets", []) or []
        ),
    })


def classify_user_message_node(state: SQLAgentState) -> SQLAgentState:
    latest = state.get("latest_user_message") or ""
    classifier_input = {
        "latest_user_message": latest,
        "request_status": state.get("request_status") or "idle",
        "active_request_type": state.get("active_request_type"),
        "missing_slots": state.get("missing_slots", []),
        "active_request_text": state.get("active_request_text"),
        "collected_fields": _compact_collected_fields(state),
        "clarification_question": state.get("clarification_question"),
        "pending_ambiguous_message": state.get("pending_ambiguous_message"),
        "recent_turns": _compact_recent_turns(state),
    }
    record_evaluation_event("classifier_input", {"input": classifier_input})

    if _is_exact_cancellation(latest):
        update = cast(SQLAgentState, {
            "user_message_type": "reset_request",
            "classification_reason": "Exact unambiguous cancellation command.",
            "active_request_action": "cancel",
            "classification_valid": True,
            "classification_error": None,
            "model_classification": None,
            "classification_safeguard": "exact_cancellation_command",
            "correction_targets": [],
            "classified_request_type": None,
        })
        record_evaluation_event("classifier_output", {
            "raw": "unavailable: deterministic exact-command classification",
            "parsed": dict(update), "parsing_error": None,
        })
        return update

    ambiguity_resolution = (
        _exact_ambiguity_resolution(latest)
        if state.get("pending_ambiguous_message")
        else None
    )
    if ambiguity_resolution:
        if (
            ambiguity_resolution == "new_request_from_ambiguous"
            and state.get("pending_ambiguous_request_type") not in {"data", "chart"}
        ):
            update = _invalid_update(
                state,
                "Saved ambiguous request has no validated request type.",
                None,
            )
            record_evaluation_event("classifier_output", {
                "raw": "unavailable: deterministic exact-choice classification",
                "parsed": None,
                "canonical": dict(update),
                "parsing_error": update.get("classification_error"),
                "safeguard": "exact_ambiguity_resolution_rejected",
            })
            return update
        update = cast(SQLAgentState, {
            "user_message_type": "clarification_or_update",
            "classification_reason": "Exact offered ambiguity resolution.",
            "active_request_action": ambiguity_resolution,
            "classification_valid": True,
            "classification_error": None,
            "model_classification": None,
            "classification_safeguard": "exact_ambiguity_resolution",
            "correction_targets": [],
            "classified_request_type": state.get("pending_ambiguous_request_type"),
        })
        record_evaluation_event("classifier_output", {
            "raw": "unavailable: deterministic exact-choice classification",
            "parsed": None,
            "canonical": dict(update),
            "parsing_error": None,
            "safeguard": "exact_ambiguity_resolution",
        })
        return update

    observable = is_evaluation_observability_enabled()
    pending_bare_date = _is_pending_bare_date_answer(state, latest)
    pending_exact_location = _is_pending_exact_location_answer(state, latest)
    raw_result = None
    parsed_candidate = None
    parsing_error = None
    raw_response = None
    model_classification = None
    try:
        chain = get_message_classification_chain(include_raw=True) if observable else get_message_classification_chain()
        raw_result = chain.invoke(classifier_input)
        if observable and isinstance(raw_result, dict) and "parsed" in raw_result:
            parsed_candidate = raw_result.get("parsed")
            parsing_error = raw_result.get("parsing_error")
            raw_response = raw_result.get("raw")
        else:
            parsed_candidate = raw_result
            raw_response = "unavailable: structured-output integration returned parsed output only"
        if parsing_error is not None:
            raise ValueError(str(parsing_error))
        model_result = MessageClassificationOutput.model_validate(parsed_candidate)
        model_classification = model_result.model_dump()
        if pending_bare_date:
            result = _canonicalize_pending_bare_date(model_result, state)
            classification_safeguard = "pending_bare_date_clarification"
        elif pending_exact_location and model_result.message_type in {
            "data_request", "clarification_or_update", "unclear",
        }:
            result = _canonicalize_pending_location(model_result, state)
            classification_safeguard = "pending_schema_location_clarification"
        elif _is_idle_request_initialization(model_result, state):
            result = _canonicalize_idle_request(model_result)
            classification_safeguard = "idle_request_initialization"
        else:
            result = model_result
            classification_safeguard = None
        coherence_error = _classification_error(result, state)
        if coherence_error:
            raise ValueError(coherence_error)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        record_evaluation_event("classifier_output", {
            "raw": compact_model_value(raw_response if raw_response is not None else raw_result),
            "parsed": compact_model_value(parsed_candidate),
            "parsing_error": error,
        })
        if model_classification is None:
            compact_candidate = compact_model_value(parsed_candidate)
            model_classification = (
                compact_candidate if isinstance(compact_candidate, dict) else None
            )
        update = _invalid_update(state, error, model_classification)
        debug_node("MESSAGE CLASSIFICATION INVALID", {"input": classifier_input, "error": error})
        return update

    correction_targets = list(result.correction_targets)
    if result.active_request_action == "continue_ambiguous":
        # Ambiguous target guesses are not validated corrections. The saved
        # message is replayed and remapped before any field replacement occurs.
        correction_targets = []

    update = cast(SQLAgentState, {
        "user_message_type": result.message_type,
        "classification_reason": result.reason,
        "active_request_action": result.active_request_action,
        "classification_valid": True,
        "classification_error": None,
        "model_classification": model_classification,
        "classification_safeguard": classification_safeguard,
        "correction_targets": correction_targets,
        "classified_request_type": result.resolved_request_type,
        "pending_ambiguous_request_type": (
            result.resolved_request_type if result.active_request_action == "ambiguous"
            else state.get("pending_ambiguous_request_type")
        ),
        "pending_ambiguous_correction_targets": (
            list(result.correction_targets) if result.active_request_action == "ambiguous"
            else state.get("pending_ambiguous_correction_targets", [])
        ),
    })
    record_evaluation_event("classifier_output", {
        "raw": compact_model_value(raw_response),
        "parsed": model_classification,
        "canonical": result.model_dump(),
        "parsing_error": None,
        "safeguard": classification_safeguard,
    })
    debug_node("MESSAGE CLASSIFICATION", {**classifier_input, **result.model_dump()})
    return update
