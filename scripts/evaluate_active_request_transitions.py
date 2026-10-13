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


STUB_LABEL = "EVALUATION_STUB_NO_DATABASE_EXECUTION"
TURN_TIMEOUT_SECONDS = 150
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = (
    PROJECT_ROOT
    / "evaluations"
    / "active_request_transition_smoke_v2_2026-10-11.md"
)

EXPECTED_SCENARIOS = [
    {
        "name": "missing_date_then_bare_date",
        "expected": (
            "The bare date is classified as a clarification answer; the original "
            "average-flow/Marconi request remains the mapping input and the generated "
            "SQL uses the explicit historical date."
        ),
        "turns": ["Show average flow in Marconi.", "2026-05-10"],
    },
    {
        "name": "missing_location_then_location",
        "expected": (
            "The location-only reply is classified as a clarification answer and adds "
            "Marconi without losing average flow or the explicit date."
        ),
        "turns": ["Show average flow on 2026-05-10.", "Marconi"],
    },
    {
        "name": "explicit_location_correction",
        "expected": (
            "The correction targets location, removes Marconi, preserves average flow, "
            "and leaves the still-missing date clarification active."
        ),
        "turns": ["Show average flow in Marconi.", "No, use Ponte instead."],
    },
    {
        "name": "schema_side_question",
        "expected": (
            "The schema question is answered as a side question while the original "
            "request and pending date clarification remain unchanged."
        ),
        "turns": ["Show average flow in Marconi.", "Which sensors are available?"],
    },
    {
        "name": "independent_new_request",
        "expected": (
            "The second message starts a fresh text request for flow in Ponte; stale "
            "Marconi, average aggregation, line-chart type, and clarification count are "
            "not carried over."
        ),
        "turns": [
            "Show a line chart of average flow in Marconi.",
            "Show flow in Ponte.",
        ],
    },
    {
        "name": "natural_language_cancellation",
        "expected": (
            "Natural-language cancellation clears request-scoped and clarification "
            "state while retaining the conversation messages."
        ),
        "turns": ["Show average flow in Marconi.", "Forget that."],
    },
    {
        "name": "ambiguous_then_resolution",
        "expected": (
            "The pressure message is classified as ambiguous, saved without changing "
            "the pending request, then applied as a metric change after the user chooses "
            "to continue; pressure is intentionally unsupported by the current schema."
        ),
        "turns": [
            "Show average flow in Marconi.",
            "What about pressure?",
            "Change the current request.",
        ],
    },
]

STATE_KEYS = (
    "request_status",
    "active_request_type",
    "active_request_text",
    "clarification_question",
    "clarification_count",
    "clarification_answers",
    "mapping_input_text",
    "raw_metric",
    "metric_column",
    "time_range",
    "filters",
    "aggregation",
    "group_by",
    "output_type",
    "chart_type",
    "missing_slots",
    "unsupported_terms",
    "pending_ambiguous_message",
    "pending_ambiguous_correction_targets",
    "sql_is_safe",
)


def relevant_state(state: dict[str, Any]) -> dict[str, Any]:
    return {key: state.get(key) for key in STATE_KEYS if key in state}


def json_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=str)


@contextmanager
def bounded_turn_timeout(seconds: int):
    def timeout_handler(_signum, _frame):
        raise TimeoutError(f"Graph turn exceeded {seconds} seconds")

    previous_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


class ForbiddenBigQueryClient:
    def __init__(self, *_args, **_kwargs):
        raise AssertionError("BigQuery client instantiation is forbidden in this evaluation")


def forbidden_bigquery_call(*_args, **_kwargs):
    raise AssertionError("BigQuery execution boundary was called during local evaluation")


def labelled_execution_stub(state: dict[str, Any]) -> dict[str, Any]:
    sql_query = state.get("sql_query")
    return {
        "query_result": {
            "success": True,
            "stubbed": True,
            "execution_label": STUB_LABEL,
            "sql_query": sql_query,
            "rows": [
                {
                    "execution_source": STUB_LABEL,
                    "value": 12.34,
                }
            ],
            "row_count": 1,
            "job_id": None,
            "total_bytes_processed": None,
            "error": None,
        },
        "execution_error": None,
    }


def latest_assistant_response(state: dict[str, Any]) -> str | None:
    for message in reversed(state.get("messages", [])):
        if getattr(message, "type", None) == "ai":
            return str(message.content)
    return None


def contains_filter(state: dict[str, Any], value: str) -> bool:
    return any(item.get("value") == value for item in state.get("filters", []) or [])


def evaluate_expected(
    scenario_name: str,
    turn_number: int,
    before: dict[str, Any],
    after: dict[str, Any],
    capture: dict[str, Any],
) -> tuple[bool, list[str]]:
    action = (capture.get("classification") or {}).get("active_request_action")
    lifecycle = capture.get("lifecycle") or {}
    mapping = capture.get("mapping") or {}
    checks: list[tuple[bool, str]] = []

    if scenario_name == "missing_location_then_location" and turn_number == 1:
        checks = [
            (after.get("request_status") == "collecting", "request remains collecting"),
            ("location" in (after.get("missing_slots") or []), "location is reported missing"),
            (
                "city or district" in (after.get("clarification_question") or "").lower(),
                "assistant asks for a location",
            ),
        ]
    elif scenario_name == "missing_date_then_bare_date" and turn_number == 2:
        sql = after.get("sql_query") or ""
        checks = [
            (action == "clarification_answer", "action is clarification_answer"),
            (lifecycle.get("active_request_text") == "Show average flow in Marconi.", "original request retained"),
            ("2026-05-10" in lifecycle.get("clarification_answers", []), "bare date merged"),
            ("Marconi" in sql and "2026-05-10" in sql, "SQL contains location and date"),
        ]
    elif scenario_name == "missing_location_then_location" and turn_number == 2:
        sql = after.get("sql_query") or ""
        checks = [
            (action == "clarification_answer", "action is clarification_answer"),
            (lifecycle.get("active_request_text") == "Show average flow on 2026-05-10.", "original request retained"),
            (mapping.get("aggregation") == "avg", "average aggregation preserved"),
            (contains_filter(mapping, "Marconi"), "Marconi mapped"),
            ("2026-05-10" in sql, "SQL contains original date"),
        ]
    elif scenario_name == "explicit_location_correction" and turn_number == 2:
        targets = (capture.get("classification") or {}).get("correction_targets", [])
        checks = [
            (action == "correction", "action is correction"),
            ("location" in targets, "location is targeted"),
            (contains_filter(mapping, "Ponte"), "Ponte is mapped"),
            (not contains_filter(mapping, "Marconi"), "Marconi is absent"),
            (mapping.get("aggregation") == "avg", "aggregation preserved"),
        ]
    elif scenario_name == "schema_side_question" and turn_number == 2:
        checks = [
            (action == "side_question", "action is side_question"),
            (after.get("active_request_text") == before.get("active_request_text"), "active request preserved"),
            (after.get("clarification_question") == before.get("clarification_question"), "pending clarification preserved"),
            (after.get("clarification_count") == before.get("clarification_count"), "clarification count preserved"),
        ]
    elif scenario_name == "independent_new_request" and turn_number == 2:
        checks = [
            (action == "new_request", "action is new_request"),
            (lifecycle.get("active_request_text") == "Show flow in Ponte.", "new request text installed"),
            (lifecycle.get("clarification_answers") == [], "old clarification answers cleared"),
            (lifecycle.get("clarification_count") == 0, "clarification count reset"),
            (lifecycle.get("aggregation") is None, "old aggregation cleared"),
            (lifecycle.get("chart_type") is None, "old chart type cleared"),
            (lifecycle.get("filters") == [], "old filters cleared before remapping"),
        ]
    elif scenario_name == "natural_language_cancellation" and turn_number == 2:
        checks = [
            (action == "cancel", "action is cancel"),
            (after.get("request_status") == "idle", "request is idle"),
            (after.get("active_request_text") is None, "active request cleared"),
            (after.get("clarification_question") is None, "clarification cleared"),
            (after.get("clarification_count") == 0, "clarification count reset"),
            (len(after.get("messages", [])) > len(before.get("messages", [])), "conversation messages retained"),
        ]
    elif scenario_name == "ambiguous_then_resolution" and turn_number == 2:
        checks = [
            (action == "ambiguous", "action is ambiguous"),
            (after.get("pending_ambiguous_message") == "What about pressure?", "ambiguous message saved"),
            (after.get("active_request_text") == before.get("active_request_text"), "pending request unchanged"),
        ]
    elif scenario_name == "ambiguous_then_resolution" and turn_number == 3:
        checks = [
            (action == "continue_ambiguous", "action is continue_ambiguous"),
            ("What about pressure?" in lifecycle.get("clarification_answers", []), "saved message applied"),
            ("Change the current request." not in lifecycle.get("clarification_answers", []), "resolution text not used as data"),
            (lifecycle.get("pending_ambiguous_message") is None, "pending ambiguity cleared"),
            (mapping.get("raw_metric") == "pressure", "pressure becomes requested metric"),
            ("pressure" in [str(item).lower() for item in mapping.get("unsupported_terms", [])], "unsupported metric reported"),
        ]
    else:
        checks = [(True, "setup turn; observed state recorded")]

    failed = [description for passed, description in checks if not passed]
    return not failed, failed or [description for passed, description in checks if passed]


def validate_explicit_configuration(settings) -> dict[str, Any]:
    role_models = {
        "interpret": settings.interpret_model,
        "mapping": settings.mapping_model,
        "clarification": settings.clarification_model,
        "sql": settings.sql_model,
        "answer": settings.answer_model,
    }
    expected_model = os.environ["EVAL_EXPECTED_MODEL"]
    if any(model != expected_model for model in role_models.values()):
        raise RuntimeError(f"Role model configuration does not match {expected_model}: {role_models}")
    if settings.langsmith_tracing:
        raise RuntimeError("External tracing must be disabled")
    return role_models


def build_report(
    role_models: dict[str, str],
    results: list[dict[str, Any]],
    started_at: str,
    total_elapsed: float,
) -> str:
    passed_turns = sum(1 for item in results if item.get("passed") is True)
    failed_turns = sum(1 for item in results if item.get("passed") is False)
    failed_scenarios = {
        item["scenario"] for item in results if item.get("passed") is False
    }
    passed_scenarios = len(EXPECTED_SCENARIOS) - len(failed_scenarios)
    lines = [
        "# Active-request transition real-Ollama smoke evaluation, v2",
        "",
        f"- Started: `{started_at}`",
        f"- Total elapsed: `{total_elapsed:.3f}s`",
        f"- Database execution: `{STUB_LABEL}` (no database calls)",
        f"- Turn timeout: `{TURN_TIMEOUT_SECONDS}s`",
        f"- Outcome: `{passed_turns} passed, {failed_turns} failed/error turns`",
        f"- Scenario outcome: `{passed_scenarios} passed, {len(failed_scenarios)} failed`",
        "- External tracing: disabled",
        "- Working directory: `/private/tmp` (project `.env` was not loaded)",
        "- Scenario contract: `evaluations/active_request_transition_scenarios_v2.md`",
        "",
        "## Exact command and configuration",
        "",
        "```bash",
        os.environ.get("EVAL_EXACT_COMMAND", "<not supplied>"),
        "```",
        "",
        "Role models:",
        "",
    ]
    for role, model in role_models.items():
        lines.append(f"- `{role}`: `{model}`")

    lines.extend(["", "## Predefined expectations", ""])
    for scenario in EXPECTED_SCENARIOS:
        lines.append(f"- **{scenario['name']}**: {scenario['expected']}")

    lines.extend(["", "## Scenario summary", ""])
    for scenario in EXPECTED_SCENARIOS:
        scenario_results = [
            item for item in results if item["scenario"] == scenario["name"]
        ]
        scenario_passed = all(item.get("passed") for item in scenario_results)
        failed_assessments = [
            detail
            for item in scenario_results
            if not item.get("passed")
            for detail in item.get("assessment", [])
        ]
        detail = "; ".join(failed_assessments) if failed_assessments else "all expectations met"
        lines.append(
            f"- **{scenario['name']} — {'PASS' if scenario_passed else 'FAIL'}:** {detail}"
        )

    lines.extend(["", "## Per-turn observations", ""])
    for item in results:
        lines.extend(
            [
                f"### {item['scenario']} — turn {item['turn']}",
                "",
                f"- User: `{item['user_message']}`",
                f"- Elapsed: `{item['elapsed_seconds']:.3f}s`",
                f"- Expected vs observed: **{'PASS' if item.get('passed') else 'FAIL'}** — "
                + "; ".join(item.get("assessment", [])),
                f"- Error: `{item.get('error')}`" if item.get("error") else "- Error: none",
                "- Structured classification:",
                "```json",
                json_text(item.get("classification")),
                "```",
                "- Evaluation observability:",
                "```json",
                json_text(item.get("observability")),
                "```",
                "- Relevant state before:",
                "```json",
                json_text(item.get("before")),
                "```",
                "- Relevant state after:",
                "```json",
                json_text(item.get("after")),
                "```",
                "- Lifecycle update:",
                "```json",
                json_text(item.get("lifecycle")),
                "```",
                "- Mapping update:",
                "```json",
                json_text(item.get("mapping")),
                "```",
                "- Assistant response:",
                "````text",
                item.get("assistant_response") or "<none>",
                "````",
                "- Generated SQL:",
                "```sql",
                item.get("generated_sql") or "-- none",
                "```",
                f"- Execution boundary: `{item.get('execution_label') or 'not reached'}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation limits",
            "",
            "This is a local-model smoke evaluation, not a database evaluation. The "
            "classification, evidence, mapping, schema-question, and SQL-generation "
            "chains are real Ollama calls. Any result rows are visibly labelled stub "
            "fixtures and are not BigQuery results.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    os.environ["AGENT_DEBUG"] = "false"
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()

    from app.config import settings
    from db import bigquery_client
    from google.cloud import bigquery
    import agent.graph as graph_module
    import agent.nodes.sql_execution as sql_execution_module
    from agent.utils.evaluation_observability import capture_evaluation_observability

    role_models = validate_explicit_configuration(settings)

    real_classifier: Callable = graph_module.classify_user_message_node
    real_lifecycle: Callable = graph_module.start_or_update_active_request_node
    real_mapping: Callable = graph_module.map_to_schema_node
    turn_capture: dict[str, Any] = {}

    def tracked_classifier(state):
        update = real_classifier(state)
        turn_capture["classification"] = dict(update)
        return update

    def tracked_lifecycle(state):
        update = real_lifecycle(state)
        turn_capture["lifecycle"] = dict(update)
        return update

    def tracked_mapping(state):
        update = real_mapping(state)
        turn_capture["mapping"] = dict(update)
        return update

    results: list[dict[str, Any]] = []

    with (
        patch.object(bigquery, "Client", ForbiddenBigQueryClient),
        patch.object(bigquery_client.bigquery, "Client", ForbiddenBigQueryClient),
        patch.object(bigquery_client, "execute_bigquery_select", forbidden_bigquery_call),
        patch.object(sql_execution_module, "execute_bigquery_select", forbidden_bigquery_call),
        patch.object(graph_module, "execute_sql_node", labelled_execution_stub),
        patch.object(graph_module, "classify_user_message_node", tracked_classifier),
        patch.object(graph_module, "start_or_update_active_request_node", tracked_lifecycle),
        patch.object(graph_module, "map_to_schema_node", tracked_mapping),
    ):
        graph = graph_module.build_graph(use_checkpointer=True)

        for scenario_index, scenario in enumerate(EXPECTED_SCENARIOS, start=1):
            config = {
                "configurable": {
                    "thread_id": f"ollama-transition-smoke-{scenario_index}"
                }
            }
            previous_state: dict[str, Any] = {}
            scenario_failed = False

            for turn_number, user_message in enumerate(scenario["turns"], start=1):
                turn_capture.clear()
                before = dict(previous_state)
                turn_started = time.monotonic()
                error: str | None = None
                after = dict(previous_state)
                observations: list[dict[str, Any]] = []

                if scenario_failed:
                    error = "Not run because an earlier turn in this scenario failed"
                else:
                    try:
                        with (
                            bounded_turn_timeout(TURN_TIMEOUT_SECONDS),
                            capture_evaluation_observability() as observations,
                        ):
                            after = graph.invoke(
                                {"messages": [HumanMessage(content=user_message)]},
                                config=config,
                            )
                        previous_state = after
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        scenario_failed = True

                elapsed = time.monotonic() - turn_started
                if error:
                    passed = False
                    assessment = [error]
                else:
                    passed, assessment = evaluate_expected(
                        scenario["name"], turn_number, before, after, turn_capture
                    )

                execution_label = None
                query_result = after.get("query_result") or {}
                if isinstance(query_result, dict):
                    execution_label = query_result.get("execution_label")

                results.append(
                    {
                        "scenario": scenario["name"],
                        "turn": turn_number,
                        "user_message": user_message,
                        "elapsed_seconds": elapsed,
                        "passed": passed,
                        "assessment": assessment,
                        "error": error,
                        "classification": turn_capture.get("classification"),
                        "before": relevant_state(before),
                        "after": relevant_state(after),
                        "lifecycle": turn_capture.get("lifecycle"),
                        "mapping": turn_capture.get("mapping"),
                        "assistant_response": latest_assistant_response(after),
                        "generated_sql": after.get("sql_query"),
                        "execution_label": execution_label,
                        "observability": observations,
                    }
                )

    report = build_report(
        role_models=role_models,
        results=results,
        started_at=started_at,
        total_elapsed=time.monotonic() - started,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(REPORT_PATH)
    print(
        json_text(
            {
                "passed": sum(1 for item in results if item["passed"]),
                "failed": sum(1 for item in results if not item["passed"]),
                "turns": len(results),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
