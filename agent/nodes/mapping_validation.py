from typing import Any, cast

from agent.state import SQLAgentState
from agent.utils.debug import debug_node
from tools.mapping_validation_tool import validate_mapping_tool


def validate_mapped_state_node(state: SQLAgentState) -> SQLAgentState:
    mapping = {
        "table_name": state.get("table_name"),
        "output_type": state.get("output_type"),
        "chart_type": state.get("chart_type"),
        "raw_metric": state.get("raw_metric"),
        "metric_column": state.get("metric_column"),
        "time_column": state.get("time_column"),
        "time_range": state.get("time_range"),
        "filters": state.get("filters", []),
        "selected_columns": state.get("selected_columns", []),
        "aggregation": state.get("aggregation"),
        "group_by": state.get("group_by", []),
        "order_by": state.get("order_by"),
        "limit": state.get("limit"),
        "unsupported_terms": state.get("unsupported_terms", []),
        "ambiguous_terms": state.get("ambiguous_terms", []),
    }

    validation_result = validate_mapping_tool(
        mapping=mapping,
        schema_context=state.get("schema_context") or {},
        semantic_context=state.get("semantic_context") or {},
        requirements_context=state.get("requirements_context") or {},
        public_schema_context=state.get("public_schema_context") or {},
    )

    validation_errors = validation_result["errors"]
    validation_warnings = validation_result["warnings"]

    missing_slots = list(state.get("missing_slots", []))

    if validation_errors:
        missing_slots.append("valid_mapping")

    missing_slots = list(dict.fromkeys(missing_slots))

    update = {
        "error": "; ".join([*validation_errors, *validation_warnings]) or None,
        "missing_slots": missing_slots,
    }

    debug_node(
        "MAPPING VALIDATION",
        {
            "mapping": mapping,
            "validation_errors": validation_errors,
            "validation_warnings": validation_warnings,
            "previous_missing_slots": state.get("missing_slots", []),
            "next_missing_slots": update["missing_slots"],
            "error": update["error"],
        },
    )

    return cast(SQLAgentState, update)