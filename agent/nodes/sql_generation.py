from typing import Any

from agent.state import SQLAgentState
from agent.prompts.sql_generation_prompt import SQL_GENERATION_PROMPT
from agent.utils.debug import debug_node
from agent.utils.evaluation_observability import (
    compact_model_value,
    is_evaluation_observability_enabled,
    record_evaluation_event,
)
from agent.chains.sql_generation_chain import get_sql_generation_chain
from models.llm import get_sql_llm
from models.structured_outputs import SQLGenerationOutput
from tools.normalize_time_range_tool import enforce_bigquery_time_predicate_tool
from tools.sql_output_shape_tool import (
    enforce_distinct_select_tool,
    enforce_having_conditions_tool,
)
from tools.sql_compiler_tool import compile_validated_sql_plan_tool


def generate_sql_node(state: SQLAgentState) -> SQLAgentState:
    """Compile validated request state without invoking the SQL model role."""
    sql_generation_input = _build_sql_generation_input(state)
    record_evaluation_event(
        "sql_generation_input",
        {"input": compact_model_value(sql_generation_input)},
    )
    record_evaluation_event(
        "sql_generation_boundary",
        {
            "provenance": "deterministic_compiler",
            "sql_model_call": False,
            "raw_provider_response": "not applicable: no SQL-model call",
            "formatted_messages": "not applicable: deterministic compiler",
        },
    )

    result = compile_validated_sql_plan_tool({**state, **sql_generation_input})
    errors = list(result["errors"])
    sql_query = result["sql_query"]
    explanation = (
        "Deterministically compiled from validated structured request state."
        if sql_query
        else "SQL compilation rejected the structured plan: " + "; ".join(errors)
    )
    record_evaluation_event(
        "sql_generation_output",
        {
            "provenance": "deterministic_compiler",
            "raw_provider_response": "not applicable: no SQL-model call",
            "parsed": None,
            "compilation_plan": compact_model_value(result.get("plan")),
            "compilation_errors": errors,
            "final_sql_query": sql_query,
        },
    )
    debug_node(
        "DETERMINISTIC SQL COMPILATION",
        {
            "input": sql_generation_input,
            "sql_query": sql_query,
            "errors": errors,
            "plan": result.get("plan"),
        },
    )
    return {
        "sql_query": sql_query,
        "sql_explanation": explanation,
        "sql_generation_provenance": "deterministic_compiler",
        "sql_compilation_errors": errors,
    }


def _build_sql_generation_input(state: SQLAgentState) -> dict[str, Any]:
    return {
        "table_name": state.get("table_name"),
        "selected_columns": state.get("selected_columns", []),
        "metric_column": state.get("metric_column"),
        "time_column": state.get("time_column"),
        "time_range": state.get("time_range"),
        "filters": state.get("filters", []),
        "aggregation": state.get("aggregation"),
        "group_by": state.get("group_by", []),
        "order_by": state.get("order_by"),
        "having_conditions": state.get("having_conditions", []),
        "distinct": state.get("distinct", False),
        "limit": state.get("limit"),
        "available_columns": state.get("available_columns", []),
        "requirements_context": state.get("requirements_context", {}),
    }


def generate_sql_with_llm_diagnostic_node(state: SQLAgentState) -> SQLAgentState:
    """Retained diagnostic-only SQL-model path; it is not wired into the graph."""
    chain = get_sql_generation_chain()

    sql_generation_input = _build_sql_generation_input(state)
    record_evaluation_event(
        "sql_generation_input",
        {"input": compact_model_value(sql_generation_input)},
    )
    if is_evaluation_observability_enabled():
        llm = get_sql_llm()
        model_settings = llm.model_dump()
        formatted_messages = SQL_GENERATION_PROMPT.format_messages(
            **sql_generation_input
        )
        record_evaluation_event(
            "sql_generation_boundary",
            {
                "formatted_messages": [
                    {
                        "role": getattr(message, "type", type(message).__name__),
                        "content": message.content,
                    }
                    for message in formatted_messages
                ],
                "structured_output": {
                    "method": "json_schema",
                    "include_raw": True,
                    "schema": SQLGenerationOutput.model_json_schema(),
                },
                "model_settings": compact_model_value(model_settings),
                "bound_invocation_options": {
                    "stop": model_settings.get("stop"),
                    "num_predict": model_settings.get("num_predict"),
                    "num_ctx": model_settings.get("num_ctx"),
                    "disable_streaming": model_settings.get("disable_streaming"),
                    "format": "SQLGenerationOutput JSON schema",
                },
            },
        )

    debug_node(
        "SQL GENERATION INPUT",
        {
            "table_name": sql_generation_input["table_name"],
            "selected_columns": sql_generation_input["selected_columns"],
            "metric_column": sql_generation_input["metric_column"],
            "time_column": sql_generation_input["time_column"],
            "time_range": sql_generation_input["time_range"],
            "filters": sql_generation_input["filters"],
            "aggregation": sql_generation_input["aggregation"],
            "group_by": sql_generation_input["group_by"],
            "order_by": sql_generation_input["order_by"],
            "limit": sql_generation_input["limit"],
            "having_conditions": sql_generation_input["having_conditions"],
            "distinct": sql_generation_input["distinct"],
        },
    )

    chain_result = chain.invoke(sql_generation_input)
    raw_response = None
    parsing_error = None
    parsed_result: Any = chain_result
    if (
        isinstance(chain_result, dict)
        and {"raw", "parsed", "parsing_error"}.issubset(chain_result)
    ):
        raw_response = chain_result["raw"]
        parsed_result = chain_result["parsed"]
        parsing_error = chain_result["parsing_error"]
        if parsing_error is not None:
            raise parsing_error

    result = SQLGenerationOutput.model_validate(parsed_result)
    provider_sql = result.sql_query

    after_time_predicate = enforce_bigquery_time_predicate_tool(
        sql_query=provider_sql,
        time_range=state.get("time_range"),
        time_column=state.get("time_column"),
    )

    after_having = enforce_having_conditions_tool(
        sql_query=after_time_predicate,
        having_conditions=state.get("having_conditions", []),
        aggregation=state.get("aggregation"),
        metric_column=state.get("metric_column"),
    )

    sql_query = enforce_distinct_select_tool(
        sql_query=after_having,
        distinct=bool(state.get("distinct", False)),
        aggregation=state.get("aggregation"),
    )
    record_evaluation_event(
        "sql_generation_output",
        {
            "raw_provider_response": compact_model_value(raw_response),
            "raw_chain_result": compact_model_value(chain_result),
            "parsed": result.model_dump(),
            "parsing_error": None,
            "transformations": {
                "provider_sql": provider_sql,
                "after_time_predicate": after_time_predicate,
                "after_having_conditions": after_having,
                "after_distinct": sql_query,
            },
            "final_sql_query": sql_query,
        },
    )

    debug_node(
        "GENERATE SQL",
        {
            "aggregation": state.get("aggregation"),
            "group_by": state.get("group_by", []),
            "selected_columns": state.get("selected_columns", []),
            "metric_column": state.get("metric_column"),
            "time_range": state.get("time_range"),
            "time_column": state.get("time_column"),
            "raw_sql_query": result.sql_query,
            "final_sql_query": sql_query,
            "having_conditions": state.get("having_conditions", []),
            "distinct": bool(state.get("distinct", False)),
        },
    )

    return {
        "sql_query": sql_query,
        "sql_explanation": result.explanation,
    }
