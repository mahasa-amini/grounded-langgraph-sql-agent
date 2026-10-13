from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage

from evaluations.transition_comparison_support import (
    GraphNodeTimingHandler,
    conversation_setup_checks,
    passed,
)
from scripts.evaluate_transition_model_comparison import (
    EVALUATION_LLM_RUNTIME,
    STUB_LABEL,
    TURN_TIMEOUT_SECONDS,
    bounded_timeout,
    latest_assistant_response,
    relevant_state,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OLD_REQUEST = "Show a line chart of average flow in Marconi."
NEW_REQUEST = "Show flow in Ponte."
NEW_REQUEST_CHOICE = "Start a new request."
DATE_ANSWER = "2026-05-10"
EXPLICIT_REQUEST = "Start a new request: show flow in Ponte on 2026-05-10."
EXPECTED_SQL_PARTS = ("value", "ponte", "2026-05-10", "2026-05-11")

EXPECTED_CONTRACT = {
    "scenario_a": {
        "turns": [OLD_REQUEST, NEW_REQUEST, NEW_REQUEST_CHOICE, DATE_ANSWER],
        "relationship_branch": (
            "If turn 2 saves the new message and asks the relationship question, "
            "send the exact offered new-request choice, then the date."
        ),
        "direct_branch": (
            "If turn 2 validly starts the fresh request, skip the relationship choice "
            "and send the date next."
        ),
    },
    "scenario_b": {"turns": [OLD_REQUEST, EXPLICIT_REQUEST]},
    "pass_requirements": {
        "setup": "Initial chart request is collecting with line, avg, and Marconi.",
        "replacement": (
            "Fresh lifecycle text is the Ponte request; stale chart, aggregation, "
            "location, clarification count, SQL/results, and ambiguity are cleared."
        ),
        "completion": (
            "Safe SQL contains value, Ponte, and the exclusive UTC date interval, "
            "then reaches only the labelled evaluation stub."
        ),
    },
}


def _has_filter(state: dict[str, Any], value: str) -> bool:
    return any(item.get("value") == value for item in state.get("filters", []) or [])


def _messages_contain(state: dict[str, Any], text: str) -> bool:
    return any(str(getattr(message, "content", "")) == text for message in state.get("messages", []))


def _fresh_lifecycle_checks(lifecycle: dict[str, Any], request_text: str) -> dict[str, bool]:
    return {
        "saved_or_explicit_request_used": lifecycle.get("active_request_text") == request_text,
        "fresh_data_request": lifecycle.get("active_request_type") == "data" and lifecycle.get("output_type") == "text",
        "old_chart_cleared": lifecycle.get("chart_type") is None,
        "old_aggregation_cleared": lifecycle.get("aggregation") is None,
        "old_location_cleared": lifecycle.get("filters") == [],
        "clarification_answers_reset": lifecycle.get("clarification_answers") == [],
        "clarification_count_reset": lifecycle.get("clarification_count") == 0,
        "pending_ambiguity_cleared": lifecycle.get("pending_ambiguous_message") is None,
        "old_sql_cleared": lifecycle.get("sql_query") is None,
        "old_result_cleared": lifecycle.get("query_result") is None,
    }


def _sql_checks(turn: dict[str, Any]) -> dict[str, bool]:
    after = turn.get("after_full") or {}
    sql = str(after.get("sql_query") or "")
    lowered = sql.lower()
    return {
        "sql_generated": bool(sql),
        "requested_metric": EXPECTED_SQL_PARTS[0] in lowered,
        "requested_location": EXPECTED_SQL_PARTS[1] in lowered,
        "inclusive_start": EXPECTED_SQL_PARTS[2] in lowered,
        "exclusive_end": EXPECTED_SQL_PARTS[3] in lowered,
        "old_average_not_inherited": "avg(" not in lowered,
        "sql_safety_passed": after.get("sql_is_safe") is True,
        "labelled_stub_reached": turn.get("execution_label") == STUB_LABEL,
    }


def assess_scenario_a(turns: list[dict[str, Any]], branch: str) -> dict[str, Any]:
    setup = turns[0].get("setup_checks", []) if turns else []
    setup_ok = bool(setup) and all(item["passed"] for item in setup)
    result: dict[str, Any] = {
        "branch": branch,
        "setup_passed": setup_ok,
        "relationship_clarification_turns": 1 if branch == "relationship_clarification" else 0,
        "date_clarification_turns": 1 if setup_ok else 0,
        "raw_model_accuracy": [],
        "canonical_actions": [],
        "state_preservation": {},
        "fresh_initialization": {},
        "completion": {},
        "passed": False,
    }
    for turn in turns:
        classification = turn.get("classification") or {}
        model = classification.get("model_classification")
        result["raw_model_accuracy"].append({
            "turn": turn["turn"],
            "raw_action": model.get("active_request_action") if isinstance(model, dict) else None,
            "applicable": isinstance(model, dict),
        })
        result["canonical_actions"].append({
            "turn": turn["turn"],
            "action": classification.get("active_request_action"),
            "safeguard": classification.get("classification_safeguard"),
        })

    if len(turns) < 2:
        return result
    turn_two = turns[1]
    after_two = turn_two.get("after_full") or {}
    result["state_preservation"] = {
        "old_request_preserved_while_unresolved": after_two.get("active_request_text") == OLD_REQUEST,
        "old_chart_preserved_while_unresolved": after_two.get("chart_type") == "line",
        "old_aggregation_preserved_while_unresolved": after_two.get("aggregation") == "avg",
        "old_location_preserved_while_unresolved": _has_filter(after_two, "Marconi"),
        "new_message_saved": after_two.get("pending_ambiguous_message") == NEW_REQUEST,
        "conversation_history_preserved": _messages_contain(after_two, OLD_REQUEST) and _messages_contain(after_two, NEW_REQUEST),
    }

    if branch == "relationship_clarification" and len(turns) >= 4:
        replacement_turn = turns[2]
        completion_turn = turns[3]
        lifecycle = replacement_turn.get("lifecycle") or {}
        result["fresh_initialization"] = _fresh_lifecycle_checks(lifecycle, NEW_REQUEST)
        result["fresh_initialization"].update({
            "resolution_sentence_not_replayed": lifecycle.get("active_request_text") != NEW_REQUEST_CHOICE,
            "ponte_mapped_after_replay": _has_filter(replacement_turn.get("mapping") or {}, "Ponte"),
            "marconi_not_mapped_after_replay": not _has_filter(replacement_turn.get("mapping") or {}, "Marconi"),
        })
        result["completion"] = _sql_checks(completion_turn)
        result["completion"]["date_applied_to_fresh_request"] = (
            NEW_REQUEST == (completion_turn.get("lifecycle") or {}).get("active_request_text")
            and DATE_ANSWER in ((completion_turn.get("lifecycle") or {}).get("clarification_answers") or [])
        )
        result["completion"]["conversation_history_preserved"] = all(
            _messages_contain(completion_turn.get("after_full") or {}, text)
            for text in (OLD_REQUEST, NEW_REQUEST, NEW_REQUEST_CHOICE, DATE_ANSWER)
        )
    elif branch == "direct_replacement" and len(turns) >= 3:
        replacement_turn = turns[1]
        completion_turn = turns[2]
        result["fresh_initialization"] = _fresh_lifecycle_checks(
            replacement_turn.get("lifecycle") or {}, NEW_REQUEST
        )
        result["fresh_initialization"].update({
            "ponte_mapped": _has_filter(replacement_turn.get("mapping") or {}, "Ponte"),
            "marconi_not_mapped": not _has_filter(replacement_turn.get("mapping") or {}, "Marconi"),
        })
        result["completion"] = _sql_checks(completion_turn)

    result["passed"] = (
        setup_ok
        and branch in {"relationship_clarification", "direct_replacement"}
        and bool(result["fresh_initialization"])
        and all(result["fresh_initialization"].values())
        and bool(result["completion"])
        and all(result["completion"].values())
    )
    return result


def assess_scenario_b(turns: list[dict[str, Any]]) -> dict[str, Any]:
    setup = turns[0].get("setup_checks", []) if turns else []
    setup_ok = bool(setup) and all(item["passed"] for item in setup)
    if len(turns) < 2:
        return {"setup_passed": setup_ok, "passed": False}
    turn = turns[1]
    classification = turn.get("classification") or {}
    model = classification.get("model_classification") or {}
    lifecycle = turn.get("lifecycle") or {}
    fresh = _fresh_lifecycle_checks(lifecycle, EXPLICIT_REQUEST)
    fresh.update({
        "ponte_mapped": _has_filter(turn.get("mapping") or {}, "Ponte"),
        "marconi_not_mapped": not _has_filter(turn.get("mapping") or {}, "Marconi"),
        "conversation_history_preserved": all(
            _messages_contain(turn.get("after_full") or {}, text)
            for text in (OLD_REQUEST, EXPLICIT_REQUEST)
        ),
    })
    sql = _sql_checks(turn)
    relationship_requested = (
        (turn.get("after_full") or {}).get("pending_ambiguous_message") == EXPLICIT_REQUEST
    )
    return {
        "setup_passed": setup_ok,
        "raw_model_action": model.get("active_request_action"),
        "raw_model_accuracy": model.get("active_request_action") == "new_request",
        "canonical_action": classification.get("active_request_action"),
        "safeguard": classification.get("classification_safeguard"),
        "relationship_clarification_requested": relationship_requested,
        "relationship_clarification_turns": 1 if relationship_requested else 0,
        "fresh_initialization": fresh,
        "completion": sql,
        "passed": setup_ok and not relationship_requested and all(fresh.values()) and all(sql.values()),
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Supplementary active-request replacement evaluation", "",
        f"- Run: `{payload['run_id']}`",
        f"- Model: `{payload['model']['tag']}` (`{payload['model']['digest']}`)",
        f"- Database attempts: `{payload['database_attempts']}`",
        f"- Labelled stub calls: `{payload['stub_executions']}`", "",
        f"- Total model calls: `{payload['model_calls']['total']}`",
        f"- SQL-generation node model calls: `{payload['model_calls']['sql_generation']}`", "",
    ]
    for scenario in payload["scenarios"]:
        lines.extend([
            f"## {scenario['name']}: {'PASS' if scenario['assessment'].get('passed') else 'FAIL'}",
            "", f"- Branch: `{scenario.get('branch')}`",
            f"- Turns executed: `{len(scenario['turns'])}`",
            f"- Assessment: `{json.dumps(scenario['assessment'], sort_keys=True)}`", "",
        ])
    lines.append("Complete per-turn evidence is retained in the companion JSON report.")
    return "\n".join(lines) + "\n"


def main() -> int:
    os.environ["AGENT_DEBUG"] = "false"
    run_id = os.environ["EVAL_RUN_ID"]
    model_tag = os.environ["EVAL_EXPECTED_MODEL"]
    model_slug = os.environ["EVAL_MODEL_SLUG"]
    output_base = PROJECT_ROOT / "evaluations" / f"replacement_paths_{run_id}_{model_slug}"
    progress_path = output_base.with_suffix(".progress.jsonl")

    def progress(event: dict[str, Any]) -> None:
        record = {"recorded_at": datetime.now(timezone.utc).isoformat(), **json.loads(json.dumps(event, default=str))}
        with progress_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(record, sort_keys=True), flush=True)

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
        raise RuntimeError(f"Non-identical configuration: roles={roles}, temperatures={temperatures}")
    if settings.langsmith_tracing:
        raise RuntimeError("External tracing must be disabled")

    def llm_factory(model: str, temperature: float) -> ChatOllama:
        return ChatOllama(model=model, base_url=settings.ollama_base_url, temperature=temperature, **EVALUATION_LLM_RUNTIME)

    database_attempts = 0
    stub_executions = 0
    model_call_events: list[dict[str, Any]] = []

    class ModelCallCounter(BaseCallbackHandler):
        def _record(self, run_id, item_count, metadata, kwargs):
            invocation_params = kwargs.get("invocation_params") or {}
            event = {
                "run_id": str(run_id),
                "node": (metadata or {}).get("langgraph_node"),
                "model": invocation_params.get("model"),
                "prompt_count": item_count,
            }
            model_call_events.append(event)
            progress({"event": "model_call_started", **event})

        def on_llm_start(
            self,
            serialized,
            prompts,
            *,
            run_id,
            metadata=None,
            invocation_params=None,
            **kwargs,
        ):
            kwargs["invocation_params"] = invocation_params
            self._record(run_id, len(prompts), metadata, kwargs)

        def on_chat_model_start(
            self,
            serialized,
            messages,
            *,
            run_id,
            metadata=None,
            **kwargs,
        ):
            self._record(run_id, len(messages), metadata, kwargs)

    def forbidden_database(*_args, **_kwargs):
        nonlocal database_attempts
        database_attempts += 1
        raise AssertionError("Database access forbidden during evaluation")

    def execution_stub(state):
        nonlocal stub_executions
        stub_executions += 1
        progress({"event": "labelled_stub_call", "label": STUB_LABEL, "sql": state.get("sql_query")})
        return {"query_result": {"success": True, "stubbed": True, "execution_label": STUB_LABEL, "rows": [{"execution_source": STUB_LABEL, "value": 12.34}], "row_count": 1, "sql_query": state.get("sql_query")}, "execution_error": None}

    capture: dict[str, Any] = {}
    real_classifier: Callable = graph_module.classify_user_message_node
    real_lifecycle: Callable = graph_module.start_or_update_active_request_node
    real_mapping: Callable = graph_module.map_to_schema_node

    def tracked_classifier(state):
        update = real_classifier(state); capture["classification"] = dict(update); return update

    def tracked_lifecycle(state):
        update = real_lifecycle(state); capture["lifecycle"] = dict(update); return update

    def tracked_mapping(state):
        update = real_mapping(state); capture["mapping"] = dict(update); return update

    def invoke_turn(graph, config, scenario: str, turn_number: int, message: str, before: dict[str, Any]) -> dict[str, Any]:
        capture.clear()
        observations: list[dict[str, Any]] = []
        timings: list[dict[str, Any]] = []
        error = None
        after = dict(before)
        tick = time.monotonic()
        progress({"event": "turn_started", "scenario": scenario, "turn": turn_number, "user_input": message, "before": relevant_state(before)})

        def node_sink(event: dict[str, Any]) -> None:
            progress({"event": "node_event", "scenario": scenario, "turn": turn_number, **event})

        try:
            with bounded_timeout(TURN_TIMEOUT_SECONDS), capture_evaluation_observability() as observations:
                after = graph.invoke(
                    {"messages": [HumanMessage(content=message)]},
                    config={
                        **config,
                        "callbacks": [
                            GraphNodeTimingHandler(timings, event_sink=node_sink),
                            ModelCallCounter(),
                        ],
                    },
                )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        record = {
            "turn": turn_number, "user_input": message,
            "elapsed_seconds": time.monotonic() - tick, "error": error,
            "classification": capture.get("classification"), "lifecycle": capture.get("lifecycle"),
            "mapping": capture.get("mapping"), "observability": observations,
            "stage_timings": timings, "before": relevant_state(before),
            "after": relevant_state(after), "after_full": after,
            "assistant_response": latest_assistant_response(after),
            "generated_sql": after.get("sql_query"),
            "execution_label": (after.get("query_result") or {}).get("execution_label"),
        }
        progress({"event": "turn_completed", "scenario": scenario, "turn": turn_number, "result": {key: value for key, value in record.items() if key != "after_full"}})
        return record

    scenarios: list[dict[str, Any]] = []
    started = time.monotonic()
    progress({"event": "run_started", "run_id": run_id, "expected_contract": EXPECTED_CONTRACT})
    with (
        patch.object(bigquery, "Client", forbidden_database),
        patch.object(bigquery_client.bigquery, "Client", forbidden_database),
        patch.object(bigquery_client, "execute_bigquery_select", forbidden_database),
        patch.object(sql_execution_module, "execute_bigquery_select", forbidden_database),
        patch.object(graph_module, "execute_sql_node", execution_stub),
        patch.object(graph_module, "classify_user_message_node", tracked_classifier),
        patch.object(graph_module, "start_or_update_active_request_node", tracked_lifecycle),
        patch.object(graph_module, "map_to_schema_node", tracked_mapping),
        patch.object(llm_module, "_create_ollama_llm", llm_factory),
    ):
        graph_a = graph_module.build_graph(use_checkpointer=True)
        config_a = {"configurable": {"thread_id": f"replacement-a-{run_id}"}}
        turns_a: list[dict[str, Any]] = []
        state_a: dict[str, Any] = {}
        for number, message in ((1, OLD_REQUEST), (2, NEW_REQUEST)):
            record = invoke_turn(graph_a, config_a, "scenario_a", number, message, state_a)
            turns_a.append(record); state_a = record["after_full"]
        setup_checks = [{"passed": ok, "description": description} for ok, description in conversation_setup_checks("independent_request", turns_a[0]["after_full"])]
        turns_a[0]["setup_checks"] = setup_checks
        if state_a.get("pending_ambiguous_message") == NEW_REQUEST:
            branch = "relationship_clarification"
            for number, message in ((3, NEW_REQUEST_CHOICE), (4, DATE_ANSWER)):
                record = invoke_turn(graph_a, config_a, "scenario_a", number, message, state_a)
                turns_a.append(record); state_a = record["after_full"]
        elif state_a.get("active_request_text") == NEW_REQUEST and not state_a.get("pending_ambiguous_message"):
            branch = "direct_replacement"
            record = invoke_turn(graph_a, config_a, "scenario_a", 3, DATE_ANSWER, state_a)
            turns_a.append(record); state_a = record["after_full"]
        else:
            branch = "unexpected_no_recovery"
        scenarios.append({"name": "scenario_a", "branch": branch, "turns": turns_a, "assessment": assess_scenario_a(turns_a, branch)})

        graph_b = graph_module.build_graph(use_checkpointer=True)
        config_b = {"configurable": {"thread_id": f"replacement-b-{run_id}"}}
        turns_b: list[dict[str, Any]] = []
        state_b: dict[str, Any] = {}
        for number, message in ((1, OLD_REQUEST), (2, EXPLICIT_REQUEST)):
            record = invoke_turn(graph_b, config_b, "scenario_b", number, message, state_b)
            turns_b.append(record); state_b = record["after_full"]
        turns_b[0]["setup_checks"] = [{"passed": ok, "description": description} for ok, description in conversation_setup_checks("independent_request", turns_b[0]["after_full"])]
        scenarios.append({"name": "scenario_b", "branch": "fixed_two_turn", "turns": turns_b, "assessment": assess_scenario_b(turns_b)})

    for scenario in scenarios:
        for turn in scenario["turns"]:
            turn.pop("after_full", None)
    payload = {
        "contract": "supplementary_replacement_paths_v1",
        "expected_contract": EXPECTED_CONTRACT,
        "run_id": run_id,
        "elapsed_seconds": time.monotonic() - started,
        "model": {"tag": model_tag, "id": os.environ.get("EVAL_MODEL_ID"), "digest": os.environ.get("EVAL_MODEL_DIGEST"), "quantization": os.environ.get("EVAL_MODEL_QUANTIZATION")},
        "roles": roles, "temperatures": temperatures,
        "timeout_seconds": TURN_TIMEOUT_SECONDS,
        "exact_command": os.environ.get("EVAL_EXACT_COMMAND"),
        "database_attempts": database_attempts, "stub_executions": stub_executions,
        "model_calls": {
            "total": len(model_call_events),
            "sql_generation": sum(
                event.get("node") == "generate_sql" for event in model_call_events
            ),
            "by_node": {
                node: sum(event.get("node") == node for event in model_call_events)
                for node in sorted(
                    {str(event.get("node")) for event in model_call_events}
                )
            },
            "events": model_call_events,
        },
        "scenarios": scenarios,
    }
    json_path = output_base.with_suffix(".json")
    md_path = output_base.with_suffix(".md")
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    md_path.write_text(_markdown(payload), encoding="utf-8")
    progress({"event": "run_completed", "json": str(json_path), "markdown": str(md_path), "database_attempts": database_attempts, "stub_executions": stub_executions, "results": [{"name": item["name"], "branch": item["branch"], "passed": item["assessment"].get("passed")} for item in scenarios]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
