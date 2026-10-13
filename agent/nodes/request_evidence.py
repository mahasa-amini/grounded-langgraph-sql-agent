from typing import cast

from agent.chains.request_evidence_chain import get_request_evidence_chain
from agent.nodes.request_interpretation import build_mapping_input_text
from agent.state import SQLAgentState
from agent.utils.debug import debug_node
from agent.utils.evaluation_observability import (
    compact_model_value,
    is_evaluation_observability_enabled,
    record_evaluation_event,
)
from models.structured_outputs import RequestEvidenceOutput
from tools.normalize_time_range_tool import (
    extract_single_explicit_absolute_date_tool,
    normalize_bare_date_tool,
)
from tools.metric_aggregation_normalization_tool import (
    normalize_metric_aggregation_evidence_tool,
)


def extract_request_evidence_node(state: SQLAgentState) -> SQLAgentState:
    mapping_input_text = build_mapping_input_text(state)

    observable = is_evaluation_observability_enabled()
    chain = (
        get_request_evidence_chain(include_raw=True)
        if observable
        else get_request_evidence_chain()
    )

    evidence_input = {
        "mapping_input_text": mapping_input_text,
        "latest_user_message": state.get("latest_user_message") or "",
        "active_request_type": state.get("active_request_type"),
        "output_type": state.get("output_type"),
        "missing_slots": state.get("missing_slots", []),
        "classification_reason": state.get("classification_reason") or "",
    }
    record_evaluation_event("evidence_input", {"input": evidence_input})
    try:
        raw_result = chain.invoke(evidence_input)
    except Exception as exc:
        record_evaluation_event("evidence_output", {
            "raw": "unavailable: evidence invocation failed before a response was returned",
            "parsed": None,
            "parsing_error": f"{type(exc).__name__}: {exc}",
        })
        raise

    raw_response = None
    parsing_error = None
    if observable and isinstance(raw_result, dict) and "parsed" in raw_result:
        parsed_candidate = raw_result.get("parsed")
        raw_response = raw_result.get("raw")
        parsing_error = raw_result.get("parsing_error")
    else:
        parsed_candidate = raw_result
        raw_response = "unavailable: structured-output integration returned parsed output only"

    if parsing_error is not None:
        record_evaluation_event("evidence_output", {
            "raw": compact_model_value(raw_response),
            "parsed": compact_model_value(parsed_candidate),
            "parsing_error": str(parsing_error),
        })
        raise ValueError(f"Request evidence parsing failed: {parsing_error}")

    try:
        result = RequestEvidenceOutput.model_validate(parsed_candidate)
    except Exception as exc:
        record_evaluation_event("evidence_output", {
            "raw": (
                compact_model_value(raw_response)
                if raw_response is not None
                else "unavailable: no raw evidence response was returned"
            ),
            "parsed": compact_model_value(parsed_candidate),
            "parsing_error": f"{type(exc).__name__}: {exc}",
        })
        raise

    location_phrases = [
        phrase.strip()
        for phrase in result.location_phrases
        if (
            isinstance(phrase, str)
            and phrase.strip()
            and normalize_bare_date_tool(phrase) is None
        )
    ]

    request_evidence = {
        "metric_phrase": (
            result.metric_phrase.strip()
            if isinstance(result.metric_phrase, str) and result.metric_phrase.strip()
            else None
        ),
        "location_phrases": location_phrases,
        "time_phrase": (
            result.time_phrase.strip()
            if isinstance(result.time_phrase, str) and result.time_phrase.strip()
            else None
        ),
        "aggregation_phrase": (
            result.aggregation_phrase.strip()
            if isinstance(result.aggregation_phrase, str)
            and result.aggregation_phrase.strip()
            else None
        ),
        "chart_phrase": (
            result.chart_phrase.strip()
            if isinstance(result.chart_phrase, str) and result.chart_phrase.strip()
            else None
        ),
    }

    metric_aggregation = normalize_metric_aggregation_evidence_tool(
        metric_phrase=request_evidence["metric_phrase"],
        aggregation_phrase=request_evidence["aggregation_phrase"],
        public_schema_context=state.get("public_schema_context") or {},
        requirements_context=state.get("requirements_context") or {},
    )
    request_evidence["metric_phrase"] = metric_aggregation["metric_phrase"]
    request_evidence["aggregation_phrase"] = metric_aggregation["aggregation_phrase"]
    request_evidence["aggregation_conflict"] = metric_aggregation["aggregation_conflict"]

    time_parser_result = None
    time_fallback_result = {"status": "not_attempted", "phrase": None, "parsed": None}
    time_phrase_provenance = "model" if request_evidence["time_phrase"] else None
    if (
        request_evidence["time_phrase"] is None
        and "time_range" in (state.get("missing_slots") or [])
    ):
        latest = state.get("latest_user_message") or ""
        time_parser_result = normalize_bare_date_tool(latest)
        if time_parser_result is not None:
            request_evidence["time_phrase"] = latest.strip()
            time_phrase_provenance = "bare_date_clarification_fallback"
            time_fallback_result = {
                "status": "applied_bare_date",
                "phrase": latest.strip(),
                "parsed": time_parser_result,
            }

    if (
        request_evidence["time_phrase"] is None
        and not (state.get("clarification_answers") or [])
    ):
        effective_request = state.get("active_request_text") or ""
        time_fallback_result = extract_single_explicit_absolute_date_tool(
            effective_request
        )
        if time_fallback_result.get("status") == "applied":
            request_evidence["time_phrase"] = time_fallback_result["phrase"]
            time_parser_result = time_fallback_result["parsed"]
            time_phrase_provenance = "embedded_absolute_date_fallback"

    request_evidence["time_phrase_provenance"] = time_phrase_provenance
    request_evidence["time_fallback_result"] = time_fallback_result

    record_evaluation_event("evidence_output", {
        "raw": compact_model_value(raw_response),
        "parsed": result.model_dump(),
        "canonical": request_evidence,
        "parsing_error": None,
        "bare_date_parser_result": time_parser_result,
        "time_phrase_provenance": time_phrase_provenance,
        "time_fallback_result": time_fallback_result,
    })

    update = {
        "mapping_input_text": mapping_input_text,
        "request_evidence": request_evidence,
    }

    debug_node(
        "REQUEST EVIDENCE EXTRACTION",
        {
            "mapping_input_text": mapping_input_text,
            "latest_user_message": state.get("latest_user_message"),
            "active_request_type": state.get("active_request_type"),
            "output_type": state.get("output_type"),
            "missing_slots": state.get("missing_slots", []),
            "request_evidence": request_evidence,
        },
    )

    return cast(SQLAgentState, update)
