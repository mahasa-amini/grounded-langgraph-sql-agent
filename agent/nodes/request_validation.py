from typing import Any, Literal

from agent.state import SQLAgentState
from agent.utils.debug import debug_node
from agent.utils.message_text import extract_plain_text_from_message_like_value
from tools.request_evidence_validation_tool import (
    validate_request_evidence_against_catalog_tool,
)


def pre_validate_request_evidence_node(state: SQLAgentState) -> dict[str, Any]:
    """
    Validate request evidence before calling the LLM schema mapper.
    """

    mapping_input_text = state.get("mapping_input_text") or state.get(
        "active_request_text",
        "",
    )

    request_text = extract_plain_text_from_message_like_value(
        mapping_input_text,
    )

    request_evidence = state.get("request_evidence") or {}

    aggregation_conflict = request_evidence.get("aggregation_conflict")
    if aggregation_conflict:
        return {
            "request_status": "collecting",
            "missing_slots": ["valid_mapping"],
            "unsupported_terms": [],
            "ambiguous_terms": [str(aggregation_conflict)],
            "evidence_suggestions": {},
            "error": str(aggregation_conflict),
        }

    validation_result = validate_request_evidence_against_catalog_tool(
        request_evidence=request_evidence,
        request_text=request_text,
    )

    debug_node(
        "PRE-VALIDATE REQUEST EVIDENCE",
        {
            "request_text": request_text,
            "request_evidence": request_evidence,
            "validation_result": validation_result,
        },
    )

    if validation_result.get("should_continue"):
        return {
            "missing_slots": [],
            "error": None,
            "unsupported_terms": [],
            "ambiguous_terms": [],
            "evidence_suggestions": {},
        }

    missing_slots = ["valid_mapping"]

    if validation_result.get("evidence_suggestions"):
        missing_slots = ["location_confirmation"]

    return {
        "request_status": "collecting",
        "missing_slots": missing_slots,
        "unsupported_terms": validation_result.get("unsupported_terms", []),
        "ambiguous_terms": validation_result.get("ambiguous_terms", []),
        "evidence_suggestions": validation_result.get("evidence_suggestions", {}),
        "error": validation_result.get("error"),
    }


def route_after_pre_validate_request_evidence(
    state: SQLAgentState,
) -> Literal["continue_to_mapping", "stop_for_clarification"]:
    """
    Route based on deterministic evidence pre-validation.
    """

    missing_slots = set(state.get("missing_slots", []) or [])

    if state.get("error"):
        return "stop_for_clarification"

    if missing_slots.intersection(
        {
            "valid_mapping",
            "location_confirmation",
            "unsupported_metric",
            "unsupported_location",
        }
    ):
        return "stop_for_clarification"

    return "continue_to_mapping"
