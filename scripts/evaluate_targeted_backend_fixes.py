from __future__ import annotations

import json
import os
import signal
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

from langchain_core.messages import HumanMessage

from evaluations.transition_comparison_support import (
    GraphNodeTimingHandler,
    build_controlled_fixtures,
    validate_controlled_fixture,
)
from scripts.evaluate_transition_model_comparison import (
    EVALUATION_LLM_RUNTIME,
    STUB_LABEL,
    TURN_TIMEOUT_SECONDS,
    latest_assistant_response,
    relevant_state,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_REQUEST = "Show average flow in Marconi on 2026-05-10."
CHECK_NAMES = (
    "pending_time_bare_date",
    "ambiguity_resolution",
    "complete_supported_sql",
)


def selected_check_names(requested_check: str | None) -> tuple[str, ...]:
    if requested_check and requested_check not in CHECK_NAMES:
        raise ValueError(
            f"Unknown EVAL_CHECK {requested_check!r}; expected one of {CHECK_NAMES}"
        )
    return (requested_check,) if requested_check else CHECK_NAMES


@contextmanager
def bounded_timeout(seconds: int):
    def timeout_handler(_signum, _frame):
        raise TimeoutError(f"Evaluation check exceeded {seconds} seconds")

    previous = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _observed_stage(events: list[dict[str, Any]], stage: str) -> dict[str, Any] | None:
    return next((event for event in reversed(events) if event.get("stage") == stage), None)


def _unfinished_nodes(events: list[dict[str, Any]]) -> list[str]:
    counts: dict[str, int] = {}
    for event in events:
        node = event.get("node")
        if not node:
            continue
        if event.get("event") == "start":
            counts[node] = counts.get(node, 0) + 1
        elif event.get("event") in {"complete", "error"}:
            counts[node] = max(0, counts.get(node, 0) - 1)
    return [node for node, count in counts.items() if count]


def _last_completed_node(events: list[dict[str, Any]]) -> str | None:
    completed = [event.get("node") for event in events if event.get("event") == "complete"]
    return completed[-1] if completed else None


def _date_check(
    before: dict[str, Any], after: dict[str, Any], capture: dict[str, Any], observations: list[dict[str, Any]]
) -> dict[str, Any]:
    classification = capture.get("classification") or {}
    lifecycle = capture.get("lifecycle") or {}
    model_classification = classification.get("model_classification") or {}
    evidence_event = _observed_stage(observations, "evidence_output") or {}
    evidence = evidence_event.get("canonical") or evidence_event.get("parsed") or {}
    normalized = (
        (capture.get("mapping") or {}).get("time_range")
        or lifecycle.get("time_range")
        or after.get("time_range")
    )
    normalized_ok = normalized == {
        "kind": "absolute",
        "relative": None,
        "start": "2026-05-10 00:00:00+00",
        "end": "2026-05-11 00:00:00+00",
    }
    lifecycle_preserved = (
        lifecycle.get("active_request_text") == before.get("active_request_text")
        and lifecycle.get("raw_metric", before.get("raw_metric")) == before.get("raw_metric")
        and lifecycle.get("aggregation", before.get("aggregation")) == before.get("aggregation")
        and lifecycle.get("filters", before.get("filters")) == before.get("filters")
        and "2026-05-10" in (lifecycle.get("clarification_answers") or [])
    )
    location_values = evidence.get("location_phrases") or []
    unsupported = after.get("unsupported_terms") or []
    return {
        "raw_model_action": model_classification.get("active_request_action"),
        "canonical_action": classification.get("active_request_action"),
        "safeguard": classification.get("classification_safeguard"),
        "raw_model_accuracy": model_classification.get("active_request_action") == "clarification_answer",
        "safeguard_applied": classification.get("classification_safeguard") == "pending_bare_date_clarification",
        "assertions": {
            "fixture_context_preserved": lifecycle_preserved,
            "date_normalized_with_exclusive_end": normalized_ok,
            "evidence_boundary_exercised": bool(evidence_event),
            "date_not_location_evidence": "2026-05-10" not in location_values,
            "date_not_unsupported": "2026-05-10" not in unsupported,
        },
        "passed": lifecycle_preserved and normalized_ok and bool(evidence_event) and "2026-05-10" not in location_values and "2026-05-10" not in unsupported,
    }


def _ambiguity_check(
    before: dict[str, Any], after: dict[str, Any], capture: dict[str, Any], observations: list[dict[str, Any]]
) -> dict[str, Any]:
    classification = capture.get("classification") or {}
    lifecycle = capture.get("lifecycle") or {}
    answers = lifecycle.get("clarification_answers") or []
    saved = before.get("pending_ambiguous_message")
    resolution = "Change the current request."
    replayed = saved in answers and resolution not in answers
    context_preserved = (
        lifecycle.get("active_request_text") == before.get("active_request_text")
        and lifecycle.get("aggregation", before.get("aggregation")) == before.get("aggregation")
        and lifecycle.get("filters", before.get("filters")) == before.get("filters")
    )
    ambiguity_cleared = lifecycle.get("pending_ambiguous_message") is None
    target_safe = classification.get("correction_targets") == [] and lifecycle.get("correction_targets") == []
    unsupported = after.get("unsupported_terms") or []
    return {
        "raw_model_action": None,
        "canonical_action": classification.get("active_request_action"),
        "safeguard": classification.get("classification_safeguard"),
        "raw_model_accuracy": None,
        "raw_model_accuracy_note": "Exact offered choice is intentionally resolved deterministically; no classifier model call is made.",
        "assertions": {
            "saved_message_replayed": replayed,
            "resolution_sentence_not_applied": resolution not in answers,
            "irrelevant_correction_targets_ignored": target_safe,
            "ambiguity_cleared_after_acceptance": ambiguity_cleared and lifecycle.get("request_transition_valid") is True,
            "existing_context_preserved": context_preserved,
        },
        "unsupported_metric_handling": {
            "requested_metric": "pressure",
            "unsupported_terms": unsupported,
            "reported_unsupported": any(str(item).lower() == "pressure" for item in unsupported),
            "sql_required": False,
        },
        "passed": replayed and target_safe and ambiguity_cleared and context_preserved,
    }


def _sql_check(after: dict[str, Any], capture: dict[str, Any]) -> dict[str, Any]:
    sql = after.get("sql_query") or ""
    lowered = sql.lower()
    classification = capture.get("classification") or {}
    assertions = {
        "classification_valid": classification.get("classification_valid") is True,
        "sql_generated": bool(sql),
        "requested_metric": "value" in lowered,
        "requested_location": "marconi" in lowered,
        "requested_start_date": "2026-05-10" in lowered,
        "exclusive_end_date": "2026-05-11" in lowered,
        "requested_aggregation": "avg" in lowered,
        "sql_safety_passed": after.get("sql_is_safe") is True,
        "labelled_stub_reached": (after.get("query_result") or {}).get("execution_label") == STUB_LABEL,
    }
    return {
        "raw_model_action": (classification.get("model_classification") or {}).get("active_request_action"),
        "canonical_action": classification.get("active_request_action"),
        "safeguard": classification.get("classification_safeguard"),
        "raw_model_accuracy": (
            (classification.get("model_classification") or {}).get("active_request_action") == "new_request"
        ),
        "assertions": assertions,
        "passed": all(assertions.values()),
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Targeted backend-fix verification",
        "",
        f"- Run: `{payload['run_id']}`",
        f"- Model: `{payload['model']['tag']}` (`{payload['model']['digest']}`)",
        f"- Timeout: `{payload['timeout_seconds']}s` per check",
        f"- Database attempts: `{payload['database_attempts']}`",
        f"- Labelled stub calls: `{payload['stub_executions']}`",
        "",
        "## Results",
        "",
    ]
    for item in payload["results"]:
        status = "PASS" if item.get("assessment", {}).get("passed") else "FAIL"
        if item.get("error"):
            status = "ERROR"
        lines.extend([
            f"### {item['check']}: {status}",
            "",
            f"- Input: `{item['user_input']}`",
            f"- Elapsed: `{item['elapsed_seconds']:.3f}s`",
            f"- Error: `{item.get('error')}`",
            f"- Last completed node: `{item.get('last_completed_node')}`",
            f"- Started but unfinished nodes: `{item.get('started_unfinished_nodes')}`",
            f"- Assessment: `{json.dumps(item.get('assessment'), sort_keys=True, default=str)}`",
            "",
        ])
    lines.append("Full raw/canonical outputs, observability events, state transitions, SQL, and node timings are in the companion JSON; incremental events are in the JSONL progress log.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    os.environ["AGENT_DEBUG"] = "false"
    run_id = os.environ["EVAL_RUN_ID"]
    model_tag = os.environ["EVAL_EXPECTED_MODEL"]
    output_base = PROJECT_ROOT / "evaluations" / f"targeted_backend_fix_verification_{run_id}_llama3_2_latest"
    progress_path = output_base.with_suffix(".progress.jsonl")

    def progress(event: dict[str, Any]) -> None:
        record = {"recorded_at": datetime.now(timezone.utc).isoformat(), **_jsonable(event)}
        with progress_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(record, sort_keys=True), flush=True)

    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    selected_names = selected_check_names(os.environ.get("EVAL_CHECK"))
    progress({"event": "run_started", "run_id": run_id, "checks": selected_names})

    from app.config import settings
    from db import bigquery_client
    from google.cloud import bigquery
    from langchain_ollama import ChatOllama
    import agent.graph as graph_module
    import agent.nodes.sql_execution as sql_execution_module
    import models.llm as llm_module
    from agent.utils.evaluation_observability import capture_evaluation_observability

    roles = {name: getattr(settings, f"{name}_model") for name in ("interpret", "mapping", "clarification", "sql", "answer")}
    temperatures = {name: getattr(settings, f"{name}_temperature") for name in roles}
    if any(value != model_tag for value in roles.values()) or any(value != 0 for value in temperatures.values()):
        raise RuntimeError(f"Non-identical evaluation configuration: roles={roles}, temperatures={temperatures}")
    if settings.langsmith_tracing:
        raise RuntimeError("External tracing must be disabled")

    def evaluation_llm_factory(model: str, temperature: float) -> ChatOllama:
        return ChatOllama(model=model, base_url=settings.ollama_base_url, temperature=temperature, **EVALUATION_LLM_RUNTIME)

    database_attempts = 0
    stub_executions = 0

    def forbidden_client(*_args, **_kwargs):
        nonlocal database_attempts
        database_attempts += 1
        raise AssertionError("BigQuery client construction forbidden")

    def forbidden_execution(*_args, **_kwargs):
        nonlocal database_attempts
        database_attempts += 1
        raise AssertionError("BigQuery execution forbidden")

    def execution_stub(state):
        nonlocal stub_executions
        stub_executions += 1
        progress({"event": "labelled_stub_call", "sql": state.get("sql_query"), "label": STUB_LABEL})
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

    real_classifier: Callable = graph_module.classify_user_message_node
    real_lifecycle: Callable = graph_module.start_or_update_active_request_node
    real_mapping: Callable = graph_module.map_to_schema_node
    capture: dict[str, Any] = {}

    def tracked_classifier(state):
        update = real_classifier(state)
        capture["classification"] = dict(update)
        return update

    def tracked_lifecycle(state):
        update = real_lifecycle(state)
        capture["lifecycle"] = dict(update)
        return update

    def tracked_mapping(state):
        update = real_mapping(state)
        capture["mapping"] = dict(update)
        return update

    fixtures = build_controlled_fixtures()
    selected_fixtures = {
        name: fixtures[name]
        for name in CHECK_NAMES[:2]
        if name in selected_names
    }
    fixture_errors = {name: validate_controlled_fixture(name, fixture) for name, fixture in selected_fixtures.items()}
    progress({"event": "fixture_validation", "results": fixture_errors})
    if any(fixture_errors.values()):
        raise RuntimeError(f"Controlled fixtures invalid: {fixture_errors}")

    all_check_specs = [
        ("pending_time_bare_date", fixtures["pending_time_bare_date"]),
        ("ambiguity_resolution", fixtures["ambiguity_resolution"]),
        ("complete_supported_sql", {"state": {}, "message": SUPPORTED_REQUEST}),
    ]
    check_specs = [item for item in all_check_specs if item[0] in selected_names]
    results: list[dict[str, Any]] = []
    with (
        patch.object(bigquery, "Client", forbidden_client),
        patch.object(bigquery_client.bigquery, "Client", forbidden_client),
        patch.object(bigquery_client, "execute_bigquery_select", forbidden_execution),
        patch.object(sql_execution_module, "execute_bigquery_select", forbidden_execution),
        patch.object(graph_module, "execute_sql_node", execution_stub),
        patch.object(graph_module, "classify_user_message_node", tracked_classifier),
        patch.object(graph_module, "start_or_update_active_request_node", tracked_lifecycle),
        patch.object(graph_module, "map_to_schema_node", tracked_mapping),
        patch.object(llm_module, "_create_ollama_llm", evaluation_llm_factory),
    ):
        for name, fixture in check_specs:
            capture.clear()
            observations: list[dict[str, Any]] = []
            timings: list[dict[str, Any]] = []
            before = fixture["state"]
            if before:
                graph_input = {**before, "messages": [*before["messages"], HumanMessage(content=fixture["message"])]}
            else:
                graph_input = {"messages": [HumanMessage(content=fixture["message"])]}
            after = dict(before)
            error = None
            tick = time.monotonic()
            progress({"event": "check_started", "check": name, "user_input": fixture["message"], "before": relevant_state(before)})

            def node_sink(event: dict[str, Any], check: str = name) -> None:
                progress({"event": "node_event", "check": check, **event})

            try:
                graph = graph_module.build_graph(use_checkpointer=False)
                with bounded_timeout(TURN_TIMEOUT_SECONDS), capture_evaluation_observability() as observations:
                    after = graph.invoke(graph_input, config={"callbacks": [GraphNodeTimingHandler(timings, event_sink=node_sink)]})
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

            if name == "pending_time_bare_date":
                assessment = _date_check(before, after, capture, observations)
            elif name == "ambiguity_resolution":
                assessment = _ambiguity_check(before, after, capture, observations)
            else:
                assessment = _sql_check(after, capture)
            if error:
                assessment["passed"] = False

            item = {
                "check": name,
                "user_input": fixture["message"],
                "fixture_validation": fixture_errors.get(name),
                "elapsed_seconds": time.monotonic() - tick,
                "error": error,
                "timed_out": bool(error and error.startswith("TimeoutError:")),
                "last_completed_node": _last_completed_node(timings),
                "started_unfinished_nodes": _unfinished_nodes(timings),
                "classification": capture.get("classification"),
                "lifecycle": capture.get("lifecycle"),
                "mapping": capture.get("mapping"),
                "observability": observations,
                "stage_timings": timings,
                "before": relevant_state(before),
                "after": relevant_state(after),
                "assistant_response": latest_assistant_response(after),
                "generated_sql": after.get("sql_query"),
                "execution_label": (after.get("query_result") or {}).get("execution_label"),
                "assessment": assessment,
            }
            results.append(item)
            progress({"event": "check_completed", "check": name, "result": item})

    payload = {
        "contract": "targeted_backend_fix_verification",
        "run_id": run_id,
        "started_at": started_at,
        "elapsed_seconds": time.monotonic() - started,
        "model": {
            "tag": model_tag,
            "id": os.environ.get("EVAL_MODEL_ID"),
            "digest": os.environ.get("EVAL_MODEL_DIGEST"),
            "quantization": os.environ.get("EVAL_MODEL_QUANTIZATION"),
        },
        "roles": roles,
        "temperatures": temperatures,
        "timeout_seconds": TURN_TIMEOUT_SECONDS,
        "evaluation_llm_runtime": EVALUATION_LLM_RUNTIME,
        "supported_request": SUPPORTED_REQUEST,
        "exact_command": os.environ.get("EVAL_EXACT_COMMAND"),
        "database_attempts": database_attempts,
        "stub_executions": stub_executions,
        "results": results,
        "summary": {
            "checks": len(results),
            "passed": sum(item["assessment"].get("passed") is True for item in results),
            "errors": sum(bool(item.get("error")) for item in results),
            "timeouts": sum(item.get("timed_out") is True for item in results),
        },
    }
    json_path = output_base.with_suffix(".json")
    markdown_path = output_base.with_suffix(".md")
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    markdown_path.write_text(_markdown(payload), encoding="utf-8")
    progress({"event": "run_completed", "json": str(json_path), "markdown": str(markdown_path), "summary": payload["summary"], "database_attempts": database_attempts})
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path), "progress": str(progress_path), "summary": payload["summary"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
