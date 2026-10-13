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
    CONVERSATION_SCENARIOS,
    GraphNodeTimingHandler,
    apply_setup_gate,
    build_controlled_fixtures,
    conversation_setup_checks,
    passed,
    validate_controlled_fixture,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STUB_LABEL = "EVALUATION_STUB_NO_DATABASE_EXECUTION"
TURN_TIMEOUT_SECONDS = 150
EVALUATION_LLM_RUNTIME = {
    "num_ctx": 8192,
    "num_predict": 2048,
    "seed": 42,
    "top_k": 40,
    "top_p": 0.9,
    "repeat_penalty": 1.0,
    "reasoning": False,
}
STATE_KEYS = (
    "request_status", "active_request_type", "active_request_text",
    "clarification_question", "clarification_count", "clarification_answers",
    "mapping_input_text", "raw_metric", "metric_column", "time_range",
    "filters", "aggregation", "group_by", "output_type", "chart_type",
    "missing_slots", "unsupported_terms", "pending_ambiguous_message",
    "pending_ambiguous_correction_targets", "model_classification",
    "classification_safeguard", "sql_query", "sql_is_safe",
)
EVALUATION_TRACKS = ("conversation", "controlled", "additional")


def selected_evaluation_tracks(requested_track: str | None) -> tuple[str, ...]:
    if requested_track and requested_track not in EVALUATION_TRACKS:
        raise ValueError(
            f"Unknown EVAL_TRACK {requested_track!r}; expected one of {EVALUATION_TRACKS}"
        )
    return (requested_track,) if requested_track else EVALUATION_TRACKS


def selected_conversation_scenarios(
    requested_scenarios: str | None,
) -> tuple[tuple[str, list[str]], ...]:
    if not requested_scenarios:
        return tuple(CONVERSATION_SCENARIOS)
    requested = tuple(
        name.strip() for name in requested_scenarios.split(",") if name.strip()
    )
    available = {name: turns for name, turns in CONVERSATION_SCENARIOS}
    unknown = [name for name in requested if name not in available]
    if unknown:
        raise ValueError(
            f"Unknown EVAL_SCENARIOS {unknown!r}; expected names from {tuple(available)}"
        )
    if len(set(requested)) != len(requested):
        raise ValueError("EVAL_SCENARIOS must not contain duplicates")
    return tuple((name, available[name]) for name in requested)


def relevant_state(state: dict[str, Any]) -> dict[str, Any]:
    return {key: state.get(key) for key in STATE_KEYS if key in state}


def latest_assistant_response(state: dict[str, Any]) -> str | None:
    for message in reversed(state.get("messages", [])):
        if getattr(message, "type", None) == "ai":
            return str(message.content)
    return None


def has_filter(state: dict[str, Any], value: str) -> bool:
    return any(item.get("value") == value for item in state.get("filters", []) or [])


@contextmanager
def bounded_timeout(seconds: int):
    def timeout_handler(_signum, _frame):
        raise TimeoutError(f"Evaluation item exceeded {seconds} seconds")
    previous = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def transition_checks(
    name: str,
    turn: int,
    before: dict[str, Any],
    after: dict[str, Any],
    capture: dict[str, Any],
) -> dict[str, Any]:
    classification = capture.get("classification") or {}
    model_classification = classification.get("model_classification")
    lifecycle = capture.get("lifecycle") or {}
    mapping = capture.get("mapping") or {}
    action = classification.get("active_request_action")
    valid = classification.get("classification_valid") is True
    classification_ok = False
    transition_ok = False
    functional_ok = False
    notes: list[str] = []
    setup_evidence: list[dict[str, Any]] | None = None

    if turn == 1:
        expected_type = "chart_request" if name == "independent_request" else "data_request"
        classification_ok = valid and action == "new_request" and classification.get("user_message_type") == expected_type
        checks = conversation_setup_checks(name, after)
        setup_evidence = [
            {"passed": value, "description": description}
            for value, description in checks
        ]
        transition_ok = passed(checks)
        functional_ok = transition_ok
        notes.extend(text for ok, text in checks if not ok)
    elif name == "missing_date":
        classification_ok = valid and action == "clarification_answer"
        transition_ok = (
            lifecycle.get("active_request_text") == "Show average flow in Marconi."
            and "2026-05-10" in (lifecycle.get("clarification_answers") or [])
        )
        sql = after.get("sql_query") or ""
        functional_ok = transition_ok and "Marconi" in sql and "2026-05-10" in sql
    elif name == "missing_location":
        classification_ok = valid and action == "clarification_answer"
        transition_ok = (
            lifecycle.get("active_request_text") == "Show average flow on 2026-05-10."
            and lifecycle.get("aggregation") in {None, "avg"}
        )
        sql = after.get("sql_query") or ""
        functional_ok = transition_ok and has_filter(mapping, "Marconi") and "2026-05-10" in sql
    elif name == "location_correction":
        classification_ok = valid and action == "correction" and "location" in (classification.get("correction_targets") or [])
        transition_ok = has_filter(mapping, "Ponte") and not has_filter(mapping, "Marconi") and mapping.get("aggregation") == "avg"
        functional_ok = transition_ok and "time_range" in (after.get("missing_slots") or [])
    elif name == "schema_side_question":
        classification_ok = valid and action == "side_question"
        transition_ok = all(
            after.get(key) == before.get(key)
            for key in ("active_request_text", "clarification_question", "clarification_count", "mapping_input_text")
        )
        functional_ok = transition_ok and bool(latest_assistant_response(after))
    elif name == "independent_request":
        classification_ok = valid and action == "new_request"
        transition_ok = (
            lifecycle.get("active_request_text") == "Show flow in Ponte."
            and lifecycle.get("clarification_answers") == []
            and lifecycle.get("clarification_count") == 0
            and lifecycle.get("aggregation") is None
            and lifecycle.get("chart_type") is None
            and lifecycle.get("filters") == []
        )
        functional_ok = transition_ok and has_filter(after, "Ponte") and not has_filter(after, "Marconi") and after.get("output_type") == "text"
    elif name == "cancellation":
        classification_ok = valid and action == "cancel"
        transition_ok = (
            after.get("request_status") == "idle"
            and after.get("active_request_text") is None
            and after.get("clarification_question") is None
            and after.get("clarification_count") == 0
        )
        functional_ok = transition_ok and len(after.get("messages", [])) > len(before.get("messages", []))
    elif name == "ambiguity_resolution" and turn == 2:
        classification_ok = valid and action == "ambiguous" and classification.get("correction_targets") == ["metric"]
        transition_ok = (
            after.get("pending_ambiguous_message") == "What about pressure?"
            and after.get("active_request_text") == before.get("active_request_text")
        )
        functional_ok = transition_ok
    elif name == "ambiguity_resolution" and turn == 3:
        classification_ok = valid and action == "continue_ambiguous"
        transition_ok = (
            "What about pressure?" in (lifecycle.get("clarification_answers") or [])
            and "Change the current request." not in (lifecycle.get("clarification_answers") or [])
            and lifecycle.get("pending_ambiguous_message") is None
        )
        functional_ok = transition_ok and mapping.get("raw_metric") == "pressure"

    safe_invalid = None
    if classification.get("classification_valid") is False:
        protected = (
            "active_request_text", "mapping_input_text", "raw_metric", "metric_column",
            "filters", "aggregation", "time_range", "chart_type", "clarification_count",
        )
        safe_invalid = all(after.get(key) == before.get(key) for key in protected)
    raw_classification_ok = None
    if isinstance(model_classification, dict):
        raw_action = model_classification.get("active_request_action")
        raw_type = model_classification.get("message_type")
        raw_targets = model_classification.get("correction_targets") or []
        if turn == 1:
            expected_type = "chart_request" if name == "independent_request" else "data_request"
            raw_classification_ok = raw_action == "new_request" and raw_type == expected_type
        elif name in {"missing_date", "missing_location"}:
            raw_classification_ok = raw_action == "clarification_answer"
        elif name == "location_correction":
            raw_classification_ok = raw_action == "correction" and "location" in raw_targets
        elif name == "schema_side_question":
            raw_classification_ok = raw_action == "side_question"
        elif name == "independent_request":
            raw_classification_ok = raw_action == "new_request"
        elif name == "cancellation":
            raw_classification_ok = raw_action == "cancel"
        elif name == "ambiguity_resolution" and turn == 2:
            raw_classification_ok = raw_action == "ambiguous" and raw_targets == ["metric"]
        elif name == "ambiguity_resolution" and turn == 3:
            raw_classification_ok = raw_action == "continue_ambiguous"

    result = {
        "classification_compliance": classification_ok,
        "raw_model_interpretation_correct": raw_classification_ok,
        "raw_model_action": (
            model_classification.get("active_request_action")
            if isinstance(model_classification, dict)
            else None
        ),
        "canonical_action": action,
        "deterministic_safeguard": classification.get("classification_safeguard"),
        "state_transition_correct": transition_ok,
        "functional_completion": functional_ok,
        "safe_invalid_output": safe_invalid,
        "model_interpretation_eligible": (
            not (name == "cancellation" and turn == 2)
            and not classification.get("classification_safeguard")
        ),
        "notes": notes,
    }
    if setup_evidence is not None:
        result["setup_preconditions"] = setup_evidence
    if name == "ambiguity_resolution" and turn == 3:
        result["saved_message_replay_correct"] = transition_ok
        result["metric_support"] = {
            "requested_metric": "pressure",
            "supported": "pressure" not in (mapping.get("unsupported_terms") or []),
            "unsupported_terms": mapping.get("unsupported_terms") or [],
        }
    return result


def fixture_turn_checks(name: str, before: dict, after: dict, capture: dict) -> dict[str, Any]:
    return transition_checks(name.replace("pending_time_bare_date", "missing_date").replace("pending_location_answer", "missing_location"), 2 if name != "ambiguity_resolution" else 3, before, after, capture)


def build_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# Transition model evaluation — {payload['model']['tag']}", "",
        f"- Started: `{payload['started_at']}`", f"- Elapsed: `{payload['elapsed_seconds']:.3f}s`",
        f"- Model digest: `{payload['model']['digest']}`", f"- Quantization: `{payload['model']['quantization']}`",
        f"- Database attempts: `{payload['database_attempts']}`", f"- Stub executions: `{payload['stub_executions']}`", "",
        "## Scores", "",
    ]
    for track in EVALUATION_TRACKS:
        summary = payload["summary"].get(track, {"status": "not_run"})
        lines.append(f"- **{track}:** {json.dumps(summary, sort_keys=True)}")
    lines.extend(["", "Full per-item traces, raw outputs, and state snapshots are in the companion JSON report.", ""])
    return "\n".join(lines)


def main() -> int:
    os.environ["AGENT_DEBUG"] = "false"
    model_tag = os.environ["EVAL_EXPECTED_MODEL"]
    model_slug = os.environ["EVAL_MODEL_SLUG"]
    run_id = os.environ["EVAL_RUN_ID"]
    output_base = PROJECT_ROOT / "evaluations" / f"transition_model_comparison_{run_id}_{model_slug}"
    progress_path = output_base.with_suffix(".progress.jsonl")
    selected_tracks = selected_evaluation_tracks(os.environ.get("EVAL_TRACK"))
    selected_scenarios = selected_conversation_scenarios(
        os.environ.get("EVAL_SCENARIOS")
    )
    if os.environ.get("EVAL_SCENARIOS") and selected_tracks != ("conversation",):
        raise ValueError("EVAL_SCENARIOS requires EVAL_TRACK=conversation")

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

    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    progress({
        "event": "run_started", "run_id": run_id, "tracks": selected_tracks,
        "conversation_scenarios": [name for name, _ in selected_scenarios],
    })

    from app.config import settings
    from db import bigquery_client
    from google.cloud import bigquery
    from langchain_ollama import ChatOllama
    import models.llm as llm_module
    import agent.graph as graph_module
    import agent.nodes.sql_execution as sql_execution_module
    from agent.nodes.request_evidence import extract_request_evidence_node
    from agent.utils.evaluation_observability import capture_evaluation_observability

    roles = {name: getattr(settings, f"{name}_model") for name in ("interpret", "mapping", "clarification", "sql", "answer")}
    temperatures = {name: getattr(settings, f"{name}_temperature") for name in roles}
    if any(value != model_tag for value in roles.values()) or any(value != 0 for value in temperatures.values()):
        raise RuntimeError(f"Non-identical configuration: roles={roles}, temperatures={temperatures}")
    if settings.langsmith_tracing:
        raise RuntimeError("External tracing must be disabled")

    def evaluation_llm_factory(model: str, temperature: float) -> ChatOllama:
        return ChatOllama(
            model=model,
            base_url=settings.ollama_base_url,
            temperature=temperature,
            **EVALUATION_LLM_RUNTIME,
        )

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
        return {"query_result": {"success": True, "stubbed": True, "execution_label": STUB_LABEL, "rows": [{"execution_source": STUB_LABEL, "value": 12.34}], "row_count": 1, "sql_query": state.get("sql_query")}, "execution_error": None}

    real_classifier: Callable = graph_module.classify_user_message_node
    real_lifecycle: Callable = graph_module.start_or_update_active_request_node
    real_mapping: Callable = graph_module.map_to_schema_node
    capture: dict[str, Any] = {}
    def tracked_classifier(state):
        update = real_classifier(state); capture["classification"] = dict(update); return update
    def tracked_lifecycle(state):
        update = real_lifecycle(state); capture["lifecycle"] = dict(update); return update
    def tracked_mapping(state):
        update = real_mapping(state); capture["mapping"] = dict(update); return update

    conversation_results = []
    controlled_results = []
    additional_results = []
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
        conversation_graph = graph_module.build_graph(use_checkpointer=True)
        scenarios = selected_scenarios if "conversation" in selected_tracks else []
        for index, (name, turns) in enumerate(scenarios, 1):
            config = {"configurable": {"thread_id": f"comparison-{run_id}-{model_slug}-{index}"}}
            previous: dict[str, Any] = {}
            setup_success = False
            turn_records = []
            for turn_number, message in enumerate(turns, 1):
                capture.clear(); observations = []; stage_timings = []; error = None; after = dict(previous); before = dict(previous)
                tick = time.monotonic()
                progress({"event": "turn_started", "scenario": name, "turn": turn_number, "user_input": message, "before": relevant_state(before)})

                def node_sink(event: dict[str, Any], scenario: str = name, turn: int = turn_number) -> None:
                    progress({"event": "node_event", "scenario": scenario, "turn": turn, **event})

                try:
                    with bounded_timeout(TURN_TIMEOUT_SECONDS), capture_evaluation_observability() as observations:
                        after = conversation_graph.invoke(
                            {"messages": [HumanMessage(content=message)]},
                            config={**config, "callbacks": [GraphNodeTimingHandler(stage_timings, event_sink=node_sink)]},
                        )
                    previous = after
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                scores = transition_checks(name, turn_number, before, after, capture) if not error else {}
                if turn_number == 1:
                    setup_success = bool(scores.get("state_transition_correct"))
                transition_demonstrated = setup_success if turn_number > 1 else True
                scores = apply_setup_gate(
                    scores,
                    setup_success=setup_success,
                    transition_turn=turn_number > 1,
                )
                turn_record = {
                    "turn": turn_number, "user_input": message, "elapsed_seconds": time.monotonic() - tick,
                    "setup_success": setup_success if turn_number == 1 else None,
                    "transition_demonstrated": transition_demonstrated, "scores": scores, "error": error,
                    "classification": capture.get("classification"), "lifecycle": capture.get("lifecycle"),
                    "mapping": capture.get("mapping"), "observability": observations,
                    "stage_timings": stage_timings,
                    "before": relevant_state(before), "after": relevant_state(after),
                    "assistant_response": latest_assistant_response(after), "generated_sql": after.get("sql_query"),
                    "execution_label": (after.get("query_result") or {}).get("execution_label"),
                }
                turn_records.append(turn_record)
                progress({"event": "turn_completed", "scenario": name, "turn": turn_number, "result": turn_record})
            conversation_results.append({"scenario": name, "setup_success": setup_success, "turns": turn_records})
            progress({"event": "scenario_completed", "scenario": name, "setup_success": setup_success})

        fixtures = build_controlled_fixtures() if "controlled" in selected_tracks else {}
        fixture_errors = {name: validate_controlled_fixture(name, fixture) for name, fixture in fixtures.items()}
        invalid_fixtures = {name: errors for name, errors in fixture_errors.items() if errors}
        if invalid_fixtures:
            raise RuntimeError(f"Controlled fixtures invalid: {invalid_fixtures}")
        for index, (name, fixture) in enumerate(fixtures.items(), 1):
            capture.clear(); observations = []; stage_timings = []; error = None
            before = fixture["state"]
            graph_input = {**before, "messages": [*before["messages"], HumanMessage(content=fixture["message"])]}
            tick = time.monotonic(); after = dict(before)
            try:
                graph = graph_module.build_graph(use_checkpointer=False)
                with bounded_timeout(TURN_TIMEOUT_SECONDS), capture_evaluation_observability() as observations:
                    after = graph.invoke(
                        graph_input,
                        config={"callbacks": [GraphNodeTimingHandler(stage_timings)]},
                    )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            scores = fixture_turn_checks(name, before, after, capture) if not error else {}
            controlled_results.append({
                "check": name, "fixture_validation": fixture_errors[name], "user_input": fixture["message"],
                "elapsed_seconds": time.monotonic() - tick, "scores": scores, "error": error,
                "classification": capture.get("classification"), "lifecycle": capture.get("lifecycle"),
                "mapping": capture.get("mapping"), "observability": observations,
                "stage_timings": stage_timings,
                "before": relevant_state(before), "after": relevant_state(after),
                "assistant_response": latest_assistant_response(after), "generated_sql": after.get("sql_query"),
            })

        if "additional" in selected_tracks:
            evidence_state = {"active_request_text": "Show average flow on 2026-05-10.", "mapping_input_text": "Show average flow on 2026-05-10.", "latest_user_message": "Show average flow on 2026-05-10.", "active_request_type": "data", "output_type": "text", "missing_slots": []}
            observations = []; error = None; tick = time.monotonic(); update = {}
            try:
                with bounded_timeout(TURN_TIMEOUT_SECONDS), capture_evaluation_observability() as observations:
                    update = extract_request_evidence_node(evidence_state)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            evidence = update.get("request_evidence") or {}
            additional_results.append({"check": "missing_location_evidence", "elapsed_seconds": time.monotonic() - tick, "passed": "2026-05-10" not in (evidence.get("location_phrases") or []), "error": error, "evidence": evidence, "observability": observations})

            capture.clear(); observations = []; stage_timings = []; error = None; tick = time.monotonic(); after = {}
            try:
                graph = graph_module.build_graph(use_checkpointer=False)
                with bounded_timeout(TURN_TIMEOUT_SECONDS), capture_evaluation_observability() as observations:
                    after = graph.invoke(
                        {"messages": [HumanMessage(content="Show average flow in Marconi on 2026-05-10.")]},
                        config={"callbacks": [GraphNodeTimingHandler(stage_timings)]},
                    )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            additional_results.append({
                "check": "complete_supported_sql", "elapsed_seconds": time.monotonic() - tick,
                "passed": bool(after.get("sql_is_safe") and after.get("sql_query") and (after.get("query_result") or {}).get("execution_label") == STUB_LABEL),
                "error": error, "classification": capture.get("classification"), "lifecycle": capture.get("lifecycle"),
                "mapping": capture.get("mapping"), "observability": observations, "after": relevant_state(after),
                "stage_timings": stage_timings,
                "assistant_response": latest_assistant_response(after), "generated_sql": after.get("sql_query"),
                "execution_label": (after.get("query_result") or {}).get("execution_label"),
            })

    def score_turns(items):
        turns = [turn for item in items for turn in item.get("turns", [item])]
        eligible = [turn for turn in turns if turn.get("scores", {}).get("model_interpretation_eligible", True)]
        raw_eligible = [
            turn for turn in turns
            if turn.get("scores", {}).get("raw_model_interpretation_correct") is not None
        ]
        return {
            "items": len(items),
            "classification_eligible": len(eligible),
            "classification_pass": sum(turn.get("scores", {}).get("classification_compliance") is True for turn in eligible),
            "raw_model_interpretation_eligible": len(raw_eligible),
            "raw_model_interpretation_pass": sum(
                turn.get("scores", {}).get("raw_model_interpretation_correct") is True
                for turn in raw_eligible
            ),
            "canonical_classification_pass": sum(
                turn.get("scores", {}).get("classification_compliance") is True
                for turn in turns
            ),
            "deterministic_safeguard_turns": sum(
                bool(turn.get("scores", {}).get("deterministic_safeguard"))
                for turn in turns
            ),
            "transition_pass": sum(turn.get("scores", {}).get("state_transition_correct") is True for turn in turns),
            "functional_pass": sum(turn.get("scores", {}).get("functional_completion") is True for turn in turns),
            "safe_invalid_pass": sum(turn.get("scores", {}).get("safe_invalid_output") is True for turn in turns),
            "elapsed_seconds": sum(turn.get("elapsed_seconds", 0) for turn in turns),
        }
    conversation_summary = score_turns(conversation_results)
    conversation_summary.update({
        "setup_pass": sum(item["setup_success"] for item in conversation_results),
        "scenario_classification_pass": sum(
            item["setup_success"] and all(
                turn["scores"].get("classification_compliance") is True
                for turn in item["turns"]
                if turn["scores"].get("model_interpretation_eligible", True)
            )
            for item in conversation_results
        ),
        "scenario_transition_pass": sum(
            item["setup_success"] and all(turn["scores"].get("state_transition_correct") is True for turn in item["turns"])
            for item in conversation_results
        ),
        "scenario_functional_pass": sum(
            item["setup_success"] and all(turn["scores"].get("functional_completion") is True for turn in item["turns"])
            for item in conversation_results
        ),
    })
    summary = {
        "conversation": (
            conversation_summary
            if "conversation" in selected_tracks else {"status": "not_run"}
        ),
        "controlled": (
            score_turns(controlled_results)
            if "controlled" in selected_tracks else {"status": "not_run"}
        ),
        "additional": (
            {"items": len(additional_results), "passed": sum(item.get("passed") is True for item in additional_results), "elapsed_seconds": sum(item["elapsed_seconds"] for item in additional_results)}
            if "additional" in selected_tracks else {"status": "not_run"}
        ),
    }
    payload = {
        "contract": "transition_model_comparison_v3", "started_at": started_at,
        "elapsed_seconds": time.monotonic() - started,
        "model": {"tag": model_tag, "id": os.environ.get("EVAL_MODEL_ID"), "digest": os.environ.get("EVAL_MODEL_DIGEST"), "quantization": os.environ.get("EVAL_MODEL_QUANTIZATION")},
        "roles": roles, "temperatures": temperatures, "timeout_seconds": TURN_TIMEOUT_SECONDS,
        "evaluation_llm_runtime": EVALUATION_LLM_RUNTIME,
        "selected_tracks": selected_tracks,
        "selected_conversation_scenarios": [name for name, _ in selected_scenarios],
        "exact_command": os.environ.get("EVAL_EXACT_COMMAND"),
        "database_attempts": database_attempts, "stub_executions": stub_executions,
        "conversation_results": conversation_results, "controlled_results": controlled_results,
        "additional_results": additional_results,
        "summary": summary,
    }
    json_path = output_base.with_suffix(".json")
    md_path = output_base.with_suffix(".md")
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    md_path.write_text(build_markdown(payload), encoding="utf-8")
    progress({"event": "run_completed", "json": str(json_path), "markdown": str(md_path), "summary": payload["summary"], "database_attempts": database_attempts, "stub_executions": stub_executions})
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "progress": str(progress_path), "summary": payload["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
