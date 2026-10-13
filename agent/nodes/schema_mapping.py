from typing import cast

from agent.state import SQLAgentState, TimeRange
from agent.utils.debug import debug_node
from agent.utils.message_text import extract_plain_text_from_message_like_value
from agent.nodes.request_interpretation import build_mapping_input_text
from agent.chains.schema_mapping_chain import get_schema_mapping_chain
from models.structured_outputs import SchemaMappingOutput

from tools.comparison_condition_resolution_tool import (
    resolve_comparison_conditions_from_text,
)
from tools.chart_type_resolution_tool import resolve_chart_type_from_text
from tools.query_filter_resolution_tool import resolve_query_filters
from tools.query_projection_resolution_tool import (
    resolve_query_projection,
    apply_post_guardrail_projection_cleanup,
)
from tools.schema_mapping_payload_tool import build_pre_guardrail_mapping
from tools.aggregation_detection_tool import detect_aggregation_from_phrase
from tools.apply_request_evidence_guardrail_tool import (
    apply_request_evidence_guardrail_tool,
)

from tools.schema_context_resolution_tool import resolve_schema_table_context

from tools.filter_normalization_tool import (
    deduplicate_filters,
    normalize_multi_value_equality_filters,
)
from tools.schema_dimension_tool import (
    collect_schema_dimension_columns_from_mapping,
    detect_requested_group_by_columns,
    remove_requested_group_by_phrases_from_locations,
    remove_schema_dimension_location_phrases,
    remove_schema_dimension_unsupported_terms,
)
from tools.ranking_cleanup_tool import (
    apply_ranking_cleanup_to_mapping,
    normalize_ranking_text,
)
from tools.query_shape_tool import enforce_raw_query_shape
from tools.semantic_mapping_tool import (
    apply_semantic_mapping_resolution,
    load_semantic_mapping_config,
)
from tools.term_cleanup_tool import (
    collect_resolved_terms_for_term_cleanup,
    remove_hallucinated_or_resolved_terms,
)
from tools.time_range_resolution_tool import resolve_effective_time_range
from tools.normalize_time_range_tool import normalize_bare_date_tool
from tools.request_resolution_tool import (
    resolve_core_mapping_fields,
    resolve_order_by_and_limit,
    resolve_output_fields,
)


def _remove_recognized_bare_date_terms(
    terms: list[str],
    time_phrase: object,
) -> list[str]:
    if not isinstance(time_phrase, str) or normalize_bare_date_tool(time_phrase) is None:
        return terms
    normalized_time = " ".join(time_phrase.strip().lower().split())
    return [
        term
        for term in terms
        if " ".join(str(term).strip().lower().split()) != normalized_time
    ]

def map_to_schema_node(state: SQLAgentState) -> SQLAgentState:
    mapping_input_text = build_mapping_input_text(state)
    correction_targets = set(state.get("correction_targets", []) or [])
    latest_user_message = state.get("latest_user_message") or ""
    filter_resolution_text = (
        latest_user_message
        if correction_targets.intersection({"location", "filters"})
        else mapping_input_text
    )

    request_evidence = dict(state.get("request_evidence") or {})
    sanitized_request_evidence = dict(request_evidence)


    requested_group_by_columns = detect_requested_group_by_columns(mapping_input_text)
    sanitized_request_evidence["location_phrases"] = (
    remove_requested_group_by_phrases_from_locations(
        location_phrases=sanitized_request_evidence.get("location_phrases"),
        requested_group_by_columns=requested_group_by_columns )
        )
    aggregation_phrase_value = sanitized_request_evidence.get("aggregation_phrase")

    aggregation_phrase = (
        aggregation_phrase_value
        if isinstance(aggregation_phrase_value, str)
        else None
    )

    aggregation_from_evidence = detect_aggregation_from_phrase(aggregation_phrase)
    debug_node(
        "MAP TO SCHEMA INPUT",
        {
            "latest_user_message": state.get("latest_user_message"),
            "active_request_text": state.get("active_request_text"),
            "clarification_answers": state.get("clarification_answers", []),
            "mapping_input_text": mapping_input_text,
            "request_evidence": state.get("request_evidence"),
            "sanitized_request_evidence": sanitized_request_evidence,
            "requested_group_by_columns": requested_group_by_columns,
            "aggregation_from_evidence": aggregation_from_evidence,
            "active_request_type": state.get("active_request_type"),
            "output_type": state.get("output_type"),
            "chart_type": state.get("chart_type"),
            "request_status": state.get("request_status"),
            "missing_slots": state.get("missing_slots", []),
            "classification_reason": state.get("classification_reason"),
            "correction_targets": sorted(correction_targets),
        },
    )

    chain = get_schema_mapping_chain()

    raw_result = chain.invoke(
        {
            "mapping_input_text": mapping_input_text,
            "latest_user_message": state.get("latest_user_message") or "",
            "active_request_type": state.get("active_request_type"),
            "output_type": state.get("output_type"),
            "chart_type": state.get("chart_type"),
            "request_status": state.get("request_status") or "collecting",
            "request_evidence": sanitized_request_evidence,
            "missing_slots": state.get("missing_slots", []),
            "classification_reason": state.get("classification_reason") or "",
            "correction_targets": sorted(correction_targets),
            "schema_context": state.get("schema_context") or {},
            "semantic_context": state.get("semantic_context") or {},
            "public_schema_context": state.get("public_schema_context") or {},
            "requirements_context": state.get("requirements_context") or {},
        }
    )

    mapping = SchemaMappingOutput.model_validate(raw_result)

    evidence_time_phrase = sanitized_request_evidence.get("time_phrase")
    unsupported_terms = _remove_recognized_bare_date_terms(
        list(mapping.unsupported_terms), evidence_time_phrase
    )
    ambiguous_terms = _remove_recognized_bare_date_terms(
        list(mapping.ambiguous_terms), evidence_time_phrase
    )

    raw_metric = mapping.raw_metric

    if raw_metric is None and len(unsupported_terms) == 1:
        raw_metric = unsupported_terms[0]

    debug_node(
        "RAW SCHEMA MAPPING OUTPUT",
        {
            "model_dump": mapping.model_dump(),
            "table_name": mapping.table_name,
            "active_request_type": mapping.active_request_type,
            "output_type": mapping.output_type,
            "chart_type": mapping.chart_type,
            "raw_metric": raw_metric,
            "metric_column": mapping.metric_column,
            "time_column": mapping.time_column,
            "time_range": mapping.time_range.model_dump() if mapping.time_range else None,
            "filters": [filter_item.model_dump() for filter_item in mapping.filters],
            "selected_columns": list(mapping.selected_columns),
            "aggregation": mapping.aggregation,
            "aggregation_from_evidence": aggregation_from_evidence,
            "group_by": list(mapping.group_by),
            "requested_group_by_columns": requested_group_by_columns,
            "order_by": mapping.order_by,
            "limit": mapping.limit,
            "missing_slots_from_mapper": list(mapping.missing_slots),
            "unsupported_terms": unsupported_terms,
            "ambiguous_terms": ambiguous_terms,
        },
    )

    table_name, defaults, available_columns = resolve_schema_table_context(
        mapped_table_name=mapping.table_name,
        previous_table_name=state.get("table_name"),
        schema_context=state.get("schema_context"),
        requirements_context=state.get("requirements_context"),
    )

    active_request_type, output_type, chart_type = resolve_output_fields(
        mapped_active_request_type=mapping.active_request_type,
        previous_active_request_type=state.get("active_request_type"),
        mapped_output_type=mapping.output_type,
        previous_output_type=state.get("output_type"),
        mapped_chart_type=mapping.chart_type,
        previous_chart_type=state.get("chart_type"),
    )

    metric_column, time_column, aggregation = resolve_core_mapping_fields(
        mapped_metric_column=mapping.metric_column,
        previous_metric_column=state.get("metric_column"),
        default_metric_column=defaults.get("metric_column"),
        mapped_time_column=mapping.time_column,
        previous_time_column=state.get("time_column"),
        default_time_column=defaults.get("time_column"),
        aggregation_from_evidence=aggregation_from_evidence,
        mapped_aggregation=mapping.aggregation,
        previous_aggregation=state.get("aggregation"),
        unsupported_terms=unsupported_terms,
    )

    parsed_time_range: TimeRange | None = state.get("parsed_time_range")
    previous_time_range: TimeRange | None = state.get("time_range")

    time_range = resolve_effective_time_range(
        parsed_time_range=parsed_time_range,
        previous_time_range=previous_time_range,
        time_phrase=sanitized_request_evidence.get("time_phrase"),
    )

    mapped_filters_for_resolution = list(mapping.filters)
    if "location" in correction_targets:
        mapped_filters_for_resolution = [
            item
            for item in mapped_filters_for_resolution
            if getattr(item, "column", None) not in {"city", "district"}
        ]

    filters, multi_value_filter_columns = resolve_query_filters(
        mapped_filters=mapped_filters_for_resolution,
        previous_filters=state.get("filters"),
        mapping_input_text=filter_resolution_text,
        semantic_context=state.get("semantic_context"),
    )

    selected_columns, group_by = resolve_query_projection(
        mapped_selected_columns=mapping.selected_columns,
        mapped_group_by=mapping.group_by,
        requested_group_by_columns=requested_group_by_columns,
        available_columns=available_columns,
        aggregation=aggregation,
        filters=filters,
        multi_value_filter_columns=multi_value_filter_columns,
        unsupported_terms=unsupported_terms,
        time_column=time_column,
        metric_column=metric_column,
)

    order_by, limit = resolve_order_by_and_limit(
        mapped_order_by=mapping.order_by,
        previous_order_by=state.get("order_by"),
        aggregation=aggregation,
        group_by=group_by,
        time_column=time_column,
        mapped_limit=mapping.limit,
        previous_limit=state.get("limit"),
        default_limit=defaults.get("default_limit", 1000),
    )

    pre_guardrail_mapping = build_pre_guardrail_mapping(
        table_name=table_name,
        active_request_type=active_request_type,
        output_type=output_type,
        chart_type=chart_type,
        raw_metric=raw_metric,
        selected_columns=selected_columns,
        metric_column=metric_column,
        time_column=time_column,
        time_range=time_range,
        filters=filters,
        aggregation=aggregation,
        group_by=group_by,
        order_by=order_by,
        limit=limit,
        unsupported_terms=unsupported_terms,
        ambiguous_terms=ambiguous_terms,
    )

    dimension_columns_for_guardrail = collect_schema_dimension_columns_from_mapping(
        mapping=pre_guardrail_mapping,
        available_columns=available_columns,
    )

    guardrail_request_evidence = remove_schema_dimension_location_phrases(
        request_evidence=sanitized_request_evidence,
        dimension_columns=dimension_columns_for_guardrail,
    )

    guardrail_result = apply_request_evidence_guardrail_tool(
        mapping=pre_guardrail_mapping,
        request_evidence=guardrail_request_evidence,
        public_schema_context=state.get("public_schema_context") or {},
        semantic_context=state.get("semantic_context") or {},
        requirements_context=state.get("requirements_context") or {},
    )

    guarded_mapping = guardrail_result["mapping"]
    evidence_suggestions = guardrail_result["evidence_suggestions"]

    guarded_dimension_columns = collect_schema_dimension_columns_from_mapping(
        mapping=guarded_mapping,
        available_columns=available_columns,
    )

    guarded_mapping["unsupported_terms"] = remove_schema_dimension_unsupported_terms(
        unsupported_terms=guarded_mapping.get("unsupported_terms", []),
        dimension_columns=guarded_dimension_columns,
    )

    guarded_filters, guarded_multi_value_filter_columns = normalize_multi_value_equality_filters(
        guarded_mapping.get("filters", [])
    )
    guarded_mapping["filters"] = guarded_filters

    guarded_mapping["time_range"] = time_range

    guarded_mapping = apply_post_guardrail_projection_cleanup(
    mapping=guarded_mapping,
    available_columns=available_columns,
    multi_value_filter_columns=guarded_multi_value_filter_columns,
)


    semantic_resolution_text = (
        latest_user_message if "metric" in correction_targets else mapping_input_text
    )
    term_cleanup_request_text = extract_plain_text_from_message_like_value(
        semantic_resolution_text
    )

    semantic_mapping_config = load_semantic_mapping_config()

    guarded_mapping, semantic_resolution_debug = apply_semantic_mapping_resolution(
        mapping=guarded_mapping,
        request_text=term_cleanup_request_text,
        semantic_mapping_config=semantic_mapping_config,
        available_columns=available_columns,
    )

    resolved_terms_for_cleanup = collect_resolved_terms_for_term_cleanup(
        mapping=guarded_mapping,
        request_evidence=guardrail_request_evidence,
        available_columns=available_columns,
    )

    resolved_terms_for_cleanup.update(
        set(semantic_resolution_debug.get("resolved_terms", []))
    )

    guarded_mapping["unsupported_terms"] = remove_hallucinated_or_resolved_terms(
        terms=guarded_mapping.get("unsupported_terms", []),
        request_text=term_cleanup_request_text,
        resolved_terms=resolved_terms_for_cleanup,
    )

    guarded_mapping["ambiguous_terms"] = remove_hallucinated_or_resolved_terms(
        terms=guarded_mapping.get("ambiguous_terms", []),
        request_text=term_cleanup_request_text,
        resolved_terms=resolved_terms_for_cleanup,
    )

    guarded_mapping["unsupported_terms"] = _remove_recognized_bare_date_terms(
        guarded_mapping.get("unsupported_terms", []), evidence_time_phrase
    )
    guarded_mapping["ambiguous_terms"] = _remove_recognized_bare_date_terms(
        guarded_mapping.get("ambiguous_terms", []), evidence_time_phrase
    )

    ranking_request_text = term_cleanup_request_text

    guarded_mapping = apply_ranking_cleanup_to_mapping(
        mapping=guarded_mapping,
        request_text=ranking_request_text,
    )

    guarded_mapping = enforce_raw_query_shape(
        mapping=guarded_mapping,
        available_columns=available_columns,
    )

    comparison_resolution = resolve_comparison_conditions_from_text(
        request_text=term_cleanup_request_text,
        metric_column=guarded_mapping.get("metric_column"),
        aggregation=guarded_mapping.get("aggregation"),
        available_columns=available_columns,
    )

    where_conditions = comparison_resolution.get("where_conditions", [])
    having_conditions = comparison_resolution.get("having_conditions", [])
    requested_output_columns = comparison_resolution.get("requested_output_columns", [])
    distinct = bool(comparison_resolution.get("distinct", False))

    chart_phrase_value = sanitized_request_evidence.get("chart_phrase")
    chart_phrase = chart_phrase_value if isinstance(chart_phrase_value, str) else None

    current_chart_type_value = guarded_mapping.get("chart_type")
    current_chart_type = (
        current_chart_type_value if isinstance(current_chart_type_value, str) else None
    )

    chart_type_resolution = resolve_chart_type_from_text(
        chart_phrase=chart_phrase,
        request_text=(
            latest_user_message
            if "chart_type" in correction_targets
            else term_cleanup_request_text
        ),
        current_chart_type=current_chart_type,
    )

    if guarded_mapping.get("output_type") == "chart":
        guarded_mapping["chart_type"] = chart_type_resolution.get("chart_type")
    else:
        guarded_mapping["chart_type"] = None


    if where_conditions:
        guarded_mapping["filters"] = deduplicate_filters(
            [
                *guarded_mapping.get("filters", []),
                *where_conditions,
            ]
        )

    if requested_output_columns:
        valid_requested_output_columns = [
            column
            for column in requested_output_columns
            if column in available_columns
        ]

        if valid_requested_output_columns:
            guarded_mapping["selected_columns"] = valid_requested_output_columns

            if guarded_mapping.get("aggregation"):
                existing_group_by = [
                    column
                    for column in guarded_mapping.get("group_by", [])
                    if column in available_columns
                ]

                for column in valid_requested_output_columns:
                    if column not in existing_group_by:
                        existing_group_by.append(column)

                guarded_mapping["group_by"] = existing_group_by
                guarded_mapping["distinct"] = False
            else:
                guarded_mapping["group_by"] = []
                guarded_mapping["order_by"] = None
                guarded_mapping["distinct"] = distinct

    guarded_mapping["having_conditions"] = having_conditions
    guarded_mapping["distinct"] = bool(guarded_mapping.get("distinct", False))

    debug_node(
        "REQUEST EVIDENCE GUARDRAIL",
        {
            "request_evidence": state.get("request_evidence"),
            "sanitized_request_evidence": sanitized_request_evidence,
            "guardrail_request_evidence": guardrail_request_evidence,
            "dimension_columns_for_guardrail": sorted(dimension_columns_for_guardrail),
            "guarded_dimension_columns": sorted(guarded_dimension_columns),
            "requested_group_by_columns": requested_group_by_columns,
            "aggregation_from_evidence": aggregation_from_evidence,
            "pre_guardrail_mapping": pre_guardrail_mapping,
            "guarded_mapping": guarded_mapping,
            "evidence_suggestions": evidence_suggestions,
            "parsed_time_range": state.get("parsed_time_range"),
            "time_range_after_time_override": guarded_mapping.get("time_range"),
            "multi_value_filter_columns": sorted(multi_value_filter_columns),
            "guarded_multi_value_filter_columns": sorted(guarded_multi_value_filter_columns),
            "ranking_cleanup_order_by": guarded_mapping.get("order_by"),
            "ranking_cleanup_limit": guarded_mapping.get("limit"),
            "comparison_resolution": comparison_resolution,
            "chart_type_resolution": chart_type_resolution,
            "where_conditions": where_conditions,
            "having_conditions": having_conditions,
            "requested_output_columns": requested_output_columns,
            "distinct": guarded_mapping.get("distinct"),
            "ranking_request_text": ranking_request_text,
            "normalized_ranking_request_text": normalize_ranking_text(ranking_request_text),
            "term_cleanup_request_text": term_cleanup_request_text,
            "resolved_terms_for_cleanup": sorted(resolved_terms_for_cleanup),
            "unsupported_terms_after_term_cleanup": guarded_mapping.get("unsupported_terms"),
            "ambiguous_terms_after_term_cleanup": guarded_mapping.get("ambiguous_terms"),
            "semantic_resolution_debug": semantic_resolution_debug,
        },
    )

    update = {
        "mapping_input_text": mapping_input_text,
        "table_name": guarded_mapping.get("table_name"),
        "active_request_type": guarded_mapping.get("active_request_type"),
        "output_type": guarded_mapping.get("output_type"),
        "chart_type": guarded_mapping.get("chart_type"),
        "raw_metric": guarded_mapping.get("raw_metric"),
        "selected_columns": guarded_mapping.get("selected_columns", []),
        "metric_column": guarded_mapping.get("metric_column"),
        "time_column": guarded_mapping.get("time_column"),
        "time_range": guarded_mapping.get("time_range"),
        "filters": guarded_mapping.get("filters", []),
        "having_conditions": guarded_mapping.get("having_conditions", []),
        "distinct": bool(guarded_mapping.get("distinct", False)),
        "aggregation": guarded_mapping.get("aggregation"),
        "group_by": guarded_mapping.get("group_by", []),
        "order_by": guarded_mapping.get("order_by"),
        "limit": guarded_mapping.get("limit"),
        "missing_slots": [],
        "unsupported_terms": guarded_mapping.get("unsupported_terms", []),
        "ambiguous_terms": guarded_mapping.get("ambiguous_terms", []),
        "evidence_suggestions": evidence_suggestions,
        "error": None,
        "correction_targets": [],
    }

    debug_node(
        "MAP TO SCHEMA STATE UPDATE",
        {
            "mapping_input_text": update["mapping_input_text"],
            "table_name": update["table_name"],
            "active_request_type": update["active_request_type"],
            "output_type": update["output_type"],
            "chart_type": update["chart_type"],
            "raw_metric": update["raw_metric"],
            "metric_column": update["metric_column"],
            "time_column": update["time_column"],
            "time_range": update["time_range"],
            "filters": update["filters"],
            "having_conditions": update["having_conditions"],
            "distinct": update["distinct"],
            "selected_columns": update["selected_columns"],
            "aggregation": update["aggregation"],
            "group_by": update["group_by"],
            "order_by": update["order_by"],
            "limit": update["limit"],
            "unsupported_terms": update["unsupported_terms"],
            "ambiguous_terms": update["ambiguous_terms"],
            "evidence_suggestions": update["evidence_suggestions"],
            "error": update["error"],
        },
    )

    return cast(SQLAgentState, update)
