from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

from scripts.evaluate_transition_model_comparison import (
    EVALUATION_LLM_RUNTIME,
    STUB_LABEL,
    TURN_TIMEOUT_SECONDS,
    bounded_timeout,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_REPORT = PROJECT_ROOT / "evaluations" / (
    "replacement_paths_2026-10-12_replacement_fixes_170108_"
    "llama3_2_latest.json"
)


def _sql_input(turn: dict[str, Any]) -> dict[str, Any]:
    events = turn.get("observability") or []
    matches = [
        event.get("input")
        for event in events
        if event.get("stage") == "sql_generation_input"
    ]
    if len(matches) != 1 or not isinstance(matches[0], dict):
        raise ValueError("Expected exactly one retained SQL-generation input")
    return matches[0]


def _load_and_validate_snapshots() -> list[dict[str, Any]]:
    payload = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    scenario_a = next(item for item in payload["scenarios"] if item["name"] == "scenario_a")
    scenario_b = next(item for item in payload["scenarios"] if item["name"] == "scenario_b")
    snapshots = [
        {"name": "scenario_a_sql_snapshot", "input": _sql_input(scenario_a["turns"][-1])},
        {"name": "scenario_b_sql_snapshot", "input": _sql_input(scenario_b["turns"][-1])},
    ]
    expected_interval = {
        "kind": "absolute",
        "start": "2026-05-10 00:00:00+00",
        "end": "2026-05-11 00:00:00+00",
        "relative": None,
    }
    for snapshot in snapshots:
        value = snapshot["input"]
        checks = {
            "table": value.get("table_name") == "sensor_measurements",
            "projection": value.get("selected_columns") == ["timestamp", "value"],
            "metric": value.get("metric_column") == "value",
            "location": value.get("filters") == [
                {"column": "district", "operator": "=", "value": "Ponte"}
            ],
            "time_column": value.get("time_column") == "timestamp",
            "time_range": value.get("time_range") == expected_interval,
            "aggregation": value.get("aggregation") is None,
            "distinct_type": value.get("distinct") is False,
            "limit_type": type(value.get("limit")) is int and value.get("limit") == 1000,
            "safety_types": (
                (value.get("requirements_context") or {}).get("safety")
                == {
                    "allow_only_select": True,
                    "max_limit": 5000,
                    "require_limit": True,
                }
            ),
        }
        snapshot["fixture_validation"] = checks
        if not all(checks.values()):
            raise ValueError(f"Invalid retained SQL fixture {snapshot['name']}: {checks}")
    return snapshots


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Controlled SQL-stage replay",
        "",
        f"- Run: `{payload['run_id']}`",
        f"- Source report: `{payload['source_report']}`",
        f"- Model: `{payload['model']['tag']}` (`{payload['model']['digest']}`)",
        f"- Database attempts: `{payload['database_attempts']}`",
        f"- Labelled stub calls: `{payload['stub_executions']}`",
        "",
    ]
    for replay in payload["replays"]:
        lines.extend(
            [
                f"## {replay['name']}",
                "",
                f"- Fixture valid: `{all(replay['fixture_validation'].values())}`",
                f"- Error: `{replay['error']}`",
                f"- SQL safe: `{replay['safety'].get('sql_is_safe')}`",
                f"- Stub reached: `{replay['stub_reached']}`",
                f"- Final SQL: `{replay['generated_sql']}`",
                "",
            ]
        )
    lines.append("Exact prompts, raw provider messages, metadata, transformations, and validation errors are in the JSON report.")
    return "\n".join(lines) + "\n"


def main() -> int:
    os.environ["AGENT_DEBUG"] = "false"
    run_id = os.environ["EVAL_RUN_ID"]
    model_tag = os.environ["EVAL_EXPECTED_MODEL"]
    model_slug = os.environ["EVAL_MODEL_SLUG"]
    output_base = PROJECT_ROOT / "evaluations" / f"sql_stage_replay_{run_id}_{model_slug}"
    progress_path = output_base.with_suffix(".progress.jsonl")

    def progress(event: dict[str, Any]) -> None:
        record = {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            **json.loads(json.dumps(event, default=str)),
        }
        with progress_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(record, sort_keys=True), flush=True)

    from app.config import settings
    from db import bigquery_client
    from google.cloud import bigquery
    from langchain_ollama import ChatOllama
    import agent.nodes.sql_execution as sql_execution_module
    import models.llm as llm_module
    from agent.nodes.schema_context import load_schema_context_node
    from agent.nodes.sql_generation import generate_sql_node
    from agent.nodes.sql_safety import validate_sql_safety_node
    from agent.utils.evaluation_observability import capture_evaluation_observability

    if settings.sql_model != model_tag or settings.sql_temperature != 0:
        raise RuntimeError(
            f"Unexpected SQL role configuration: {settings.sql_model}, "
            f"temperature={settings.sql_temperature}"
        )
    if settings.langsmith_tracing:
        raise RuntimeError("External tracing must be disabled")

    snapshots = _load_and_validate_snapshots()
    database_attempts = 0
    stub_executions = 0

    def forbidden_database(*_args, **_kwargs):
        nonlocal database_attempts
        database_attempts += 1
        raise AssertionError("Database access forbidden during SQL-stage replay")

    def execution_stub(state: dict[str, Any]) -> dict[str, Any]:
        nonlocal stub_executions
        stub_executions += 1
        progress(
            {
                "event": "labelled_stub_call",
                "label": STUB_LABEL,
                "sql": state.get("sql_query"),
            }
        )
        return {
            "query_result": {
                "success": True,
                "stubbed": True,
                "execution_label": STUB_LABEL,
                "rows": [{"execution_source": STUB_LABEL, "value": 12.34}],
                "row_count": 1,
                "sql_query": state.get("sql_query"),
            },
            "execution_error": None,
        }

    def llm_factory(model: str, temperature: float) -> ChatOllama:
        return ChatOllama(
            model=model,
            base_url=settings.ollama_base_url,
            temperature=temperature,
            **EVALUATION_LLM_RUNTIME,
        )

    schema_state = load_schema_context_node({})
    replays: list[dict[str, Any]] = []
    started = time.monotonic()
    progress(
        {
            "event": "run_started",
            "run_id": run_id,
            "source_report": str(SOURCE_REPORT),
            "snapshots": [
                {"name": item["name"], "fixture_validation": item["fixture_validation"]}
                for item in snapshots
            ],
        }
    )

    with (
        patch.object(bigquery, "Client", forbidden_database),
        patch.object(bigquery_client.bigquery, "Client", forbidden_database),
        patch.object(bigquery_client, "execute_bigquery_select", forbidden_database),
        patch.object(sql_execution_module, "execute_bigquery_select", forbidden_database),
        patch.object(llm_module, "_create_ollama_llm", llm_factory),
    ):
        for snapshot in snapshots:
            tick = time.monotonic()
            observations: list[dict[str, Any]] = []
            error = None
            generated: dict[str, Any] = {}
            safety: dict[str, Any] = {}
            stub_result = None
            progress({"event": "replay_started", "name": snapshot["name"]})
            try:
                state = {**schema_state, **snapshot["input"]}
                with bounded_timeout(TURN_TIMEOUT_SECONDS), capture_evaluation_observability() as observations:
                    generated = generate_sql_node(state)
                    state = {**state, **generated}
                    safety = validate_sql_safety_node(state)
                    state = {**state, **safety}
                    if safety.get("sql_is_safe"):
                        stub_result = execution_stub(state)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            record = {
                "name": snapshot["name"],
                "fixture_validation": snapshot["fixture_validation"],
                "input": snapshot["input"],
                "elapsed_seconds": time.monotonic() - tick,
                "error": error,
                "observability": observations,
                "generated_sql": generated.get("sql_query"),
                "sql_explanation": generated.get("sql_explanation"),
                "safety": safety,
                "stub_reached": stub_result is not None,
                "stub_result": stub_result,
            }
            replays.append(record)
            progress({"event": "replay_completed", "result": record})

    payload = {
        "contract": "controlled_sql_stage_replay_v1",
        "run_id": run_id,
        "source_report": str(SOURCE_REPORT),
        "elapsed_seconds": time.monotonic() - started,
        "timeout_seconds": TURN_TIMEOUT_SECONDS,
        "exact_command": os.environ.get("EVAL_EXACT_COMMAND"),
        "model": {
            "tag": model_tag,
            "id": os.environ.get("EVAL_MODEL_ID"),
            "digest": os.environ.get("EVAL_MODEL_DIGEST"),
            "quantization": os.environ.get("EVAL_MODEL_QUANTIZATION"),
        },
        "sql_role": settings.sql_model,
        "sql_temperature": settings.sql_temperature,
        "evaluation_llm_runtime": EVALUATION_LLM_RUNTIME,
        "model_modelfile_defaults": {
            "stop": ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"],
            "num_predict": "not declared in model Modelfile",
            "num_ctx": "not declared in model Modelfile",
        },
        "database_attempts": database_attempts,
        "stub_executions": stub_executions,
        "replays": replays,
    }
    json_path = output_base.with_suffix(".json")
    md_path = output_base.with_suffix(".md")
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    md_path.write_text(_markdown(payload), encoding="utf-8")
    progress(
        {
            "event": "run_completed",
            "json": str(json_path),
            "markdown": str(md_path),
            "database_attempts": database_attempts,
            "stub_executions": stub_executions,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
