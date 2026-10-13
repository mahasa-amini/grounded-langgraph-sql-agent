"""Offline, synthetic demonstration of backend compilation and output contracts."""

from __future__ import annotations

import json
from typing import Any

from agent.nodes.chart_builder import build_chart_payload_node
from agent.nodes.response_builder import build_text_response_node
from agent.nodes.sql_safety import validate_sql_safety_node
from tools.schema_context_tool import get_schema_context_tool
from tools.sql_compiler_tool import compile_validated_sql_plan_tool


FIXTURE_LABEL = "SYNTHETIC_FIXTURE_NO_DATABASE_EXECUTION"


def build_validated_fixture() -> dict[str, Any]:
    """Return a genuinely typed plan matching the current validated contract."""
    return {
        **get_schema_context_tool(),
        "table_name": "sensor_measurements",
        "available_columns": [
            "sensor_id",
            "timestamp",
            "type",
            "value",
            "city",
            "district",
        ],
        "selected_columns": ["timestamp", "value"],
        "metric_column": "value",
        "time_column": "timestamp",
        "time_range": {
            "kind": "absolute",
            "start": "2026-05-10 00:00:00+00",
            "end": "2026-05-11 00:00:00+00",
            "relative": None,
        },
        "filters": [{"column": "district", "operator": "=", "value": "Ponte"}],
        "aggregation": None,
        "group_by": [],
        "order_by": "timestamp ASC",
        "having_conditions": [],
        "distinct": False,
        "limit": 1000,
        "unsupported_terms": [],
        "ambiguous_terms": [],
        "output_type": "text",
        "chart_type": None,
    }


def build_demo_document() -> dict[str, Any]:
    plan = build_validated_fixture()
    compilation = compile_validated_sql_plan_tool(plan)
    if compilation["errors"] or not compilation["sql_query"]:
        raise RuntimeError(f"Fixture compilation failed: {compilation['errors']}")

    sql = compilation["sql_query"]
    safety = validate_sql_safety_node({**plan, "sql_query": sql})
    if not safety["sql_is_safe"]:
        raise RuntimeError(f"Fixture SQL failed safety validation: {safety['sql_validation_errors']}")

    rows = [
        {"timestamp": "2026-05-10T00:00:00+00:00", "value": 12.34},
        {"timestamp": "2026-05-10T00:15:00+00:00", "value": 12.10},
    ]
    query_result = {
        "success": True,
        "execution_source": FIXTURE_LABEL,
        "rows": rows,
        "row_count": len(rows),
        "job_id": None,
        "total_bytes_processed": None,
        "error": None,
    }
    text_output = build_text_response_node(
        {
            **plan,
            "sql_query": sql,
            "query_result": query_result,
            "execution_error": None,
        }
    )
    chart_output = build_chart_payload_node(
        {
            **plan,
            "sql_query": sql,
            "query_result": query_result,
            "execution_error": None,
            "output_type": "chart",
            "chart_type": "line",
        }
    )

    public_plan_fields = {
        key: plan[key]
        for key in (
            "table_name",
            "selected_columns",
            "metric_column",
            "time_column",
            "time_range",
            "filters",
            "aggregation",
            "group_by",
            "order_by",
            "having_conditions",
            "distinct",
            "limit",
        )
    }
    return {
        "demo_mode": "offline fixture playback",
        "model_calls": 0,
        "database_attempts": 0,
        "conversation_fixture": [
            {"role": "user", "content": "Show flow in Ponte."},
            {"role": "assistant", "content": "For which date or time range?"},
            {"role": "user", "content": "2026-05-10"},
        ],
        "validated_plan": public_plan_fields,
        "sql_generation": {
            "provenance": "deterministic_compiler",
            "sql_model_output": "not applicable",
            "sql": sql,
            "compiler_plan": compilation["plan"],
            "safety": safety,
        },
        "execution_boundary": {
            "label": FIXTURE_LABEL,
            "synthetic": True,
            "result": query_result,
        },
        "output_contracts": {
            "text": {
                "fixture_notice": (
                    "Synthetic stub playback only; this is the deterministic "
                    "response-builder contract, not a real database result."
                ),
                "final_answer": text_output["final_answer"],
            },
            "chart_payload": chart_output["chart_payload"],
            "chart_rendered": False,
        },
    }


def main() -> None:
    print(json.dumps(build_demo_document(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
