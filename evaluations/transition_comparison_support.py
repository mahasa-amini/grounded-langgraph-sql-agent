from __future__ import annotations

from copy import deepcopy
import time
from typing import Any, Callable

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage

from agent.nodes.request_lifecycle import start_or_update_active_request_node


class GraphNodeTimingHandler(BaseCallbackHandler):
    """Evaluation-only LangGraph node timing via the callbacks API."""

    def __init__(
        self,
        events: list[dict[str, Any]],
        event_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.events = events
        self.event_sink = event_sink
        self._started: dict[Any, tuple[str, float]] = {}

    def _record(self, record: dict[str, Any]) -> None:
        self.events.append(record)
        if self.event_sink is not None:
            self.event_sink(dict(record))

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: dict[str, Any],
        *,
        run_id: Any,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        node = (metadata or {}).get("langgraph_node")
        operation = name or (serialized or {}).get("name")
        if not node or operation != node:
            return
        self._started[run_id] = (str(node), time.monotonic())
        self._record({"node": str(node), "event": "start"})

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        self._finish(run_id, "complete")

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        self._finish(run_id, "error", f"{type(error).__name__}: {error}")

    def _finish(self, run_id: Any, event: str, error: str | None = None) -> None:
        started = self._started.pop(run_id, None)
        if started is None:
            return
        node, tick = started
        record = {
            "node": node,
            "event": event,
            "elapsed_seconds": time.monotonic() - tick,
        }
        if error is not None:
            record["error"] = error
        self._record(record)


CONVERSATION_SCENARIOS = [
    ("missing_date", ["Show average flow in Marconi.", "2026-05-10"]),
    ("missing_location", ["Show average flow on 2026-05-10.", "Marconi"]),
    ("location_correction", ["Show average flow in Marconi.", "No, use Ponte instead."]),
    ("schema_side_question", ["Show average flow in Marconi.", "Which sensors are available?"]),
    ("independent_request", ["Show a line chart of average flow in Marconi.", "Show flow in Ponte."]),
    ("cancellation", ["Show average flow in Marconi.", "Forget that."]),
    ("ambiguity_resolution", ["Show average flow in Marconi.", "What about pressure?", "Change the current request."]),
]


def _base_active_request(*, chart: bool = False) -> dict[str, Any]:
    request = (
        "Show a line chart of average flow in Marconi."
        if chart else "Show average flow in Marconi."
    )
    state = {
        "latest_user_message": request,
        "user_message_type": "chart_request" if chart else "data_request",
        "active_request_action": "new_request",
        "classification_valid": True,
        "classified_request_type": "chart" if chart else "data",
        "request_status": "idle",
    }
    state.update(start_or_update_active_request_node(state))
    state.update({
        "messages": [HumanMessage(content=request), AIMessage(content="For which date?")],
        "request_status": "collecting",
        "intent": "retrieve_data",
        "active_request_type": "chart" if chart else "data",
        "output_type": "chart" if chart else "text",
        "active_request_text": request,
        "mapping_input_text": request,
        "clarification_question": "For which date?",
        "clarification_count": 1,
        "clarification_answers": [],
        "missing_slots": ["time_range"],
        "raw_metric": "flow",
        "metric_column": "value",
        "time_column": "timestamp",
        "time_range": None,
        "parsed_time_range": None,
        "filters": [{"column": "district", "operator": "=", "value": "Marconi"}],
        "aggregation": "avg",
        "group_by": ["sensor_id", "type", "city"],
        "selected_columns": ["sensor_id", "type", "city"],
        "order_by": None,
        "limit": 1000,
        "chart_type": "line" if chart else None,
        "unsupported_terms": [],
        "ambiguous_terms": [],
        "having_conditions": [],
        "distinct": False,
        "evidence_suggestions": {},
        "sql_query": None,
        "sql_is_safe": False,
        "query_result": None,
    })
    return state


def build_controlled_fixtures() -> dict[str, dict[str, Any]]:
    base = _base_active_request()
    pending_location = deepcopy(base)
    pending_location.update({
        "messages": [
            HumanMessage(content="Show average flow on 2026-05-10."),
            AIMessage(content="Which supported city or district should I use for this request?"),
        ],
        "active_request_text": "Show average flow on 2026-05-10.",
        "mapping_input_text": "Show average flow on 2026-05-10.",
        "clarification_question": "Which supported city or district should I use for this request?",
        "missing_slots": ["location"],
        "filters": [],
        "time_range": {
            "kind": "absolute", "relative": None,
            "start": "2026-05-10 00:00:00+00",
            "end": "2026-05-11 00:00:00+00",
        },
    })
    ambiguity = deepcopy(base)
    ambiguity.update({
        "messages": [
            *base["messages"], HumanMessage(content="What about pressure?"),
            AIMessage(content='Do you want to apply "What about pressure?" to the current request, or start a new request?'),
        ],
        "pending_ambiguous_message": "What about pressure?",
        "pending_ambiguous_correction_targets": ["metric"],
        "pending_ambiguous_request_type": "data",
    })
    return {
        "pending_time_bare_date": {"state": deepcopy(base), "message": "2026-05-10"},
        "pending_location_answer": {"state": pending_location, "message": "Marconi"},
        "location_correction": {"state": deepcopy(base), "message": "No, use Ponte instead."},
        "schema_side_question": {"state": deepcopy(base), "message": "Which sensors are available?"},
        "independent_request": {"state": _base_active_request(chart=True), "message": "Show flow in Ponte."},
        "cancellation": {"state": deepcopy(base), "message": "Forget that."},
        "ambiguity_resolution": {"state": ambiguity, "message": "Change the current request."},
    }


def validate_controlled_fixture(name: str, fixture: dict[str, Any]) -> list[str]:
    state = fixture.get("state") or {}
    errors: list[str] = []
    required = [
        "messages", "request_status", "active_request_type", "active_request_text",
        "mapping_input_text", "clarification_question", "clarification_count",
        "missing_slots", "raw_metric", "metric_column", "filters", "aggregation",
        "output_type",
    ]
    for key in required:
        if key not in state:
            errors.append(f"missing state key: {key}")
    if state.get("request_status") != "collecting":
        errors.append("request_status must be collecting")
    if not fixture.get("message"):
        errors.append("message is required")
    if name == "pending_time_bare_date" and "time_range" not in state.get("missing_slots", []):
        errors.append("time_range must be pending")
    if name == "pending_location_answer":
        if "location" not in state.get("missing_slots", []):
            errors.append("location must be pending")
        if state.get("filters"):
            errors.append("pending-location fixture must not contain a location filter")
    if name == "independent_request":
        if state.get("active_request_type") != "chart" or state.get("chart_type") != "line":
            errors.append("independent fixture must contain an active line-chart request")
        if state.get("aggregation") != "avg":
            errors.append("independent fixture must contain average aggregation")
        if not any(item.get("value") == "Marconi" for item in state.get("filters", [])):
            errors.append("independent fixture must contain the old Marconi location")
    if name == "ambiguity_resolution":
        if state.get("pending_ambiguous_message") != "What about pressure?":
            errors.append("saved ambiguous message is required")
        if state.get("pending_ambiguous_correction_targets") != ["metric"]:
            errors.append("saved metric correction target is required")
    return errors


def conversation_setup_checks(name: str, state: dict[str, Any]) -> list[tuple[bool, str]]:
    common = [
        (state.get("request_status") == "collecting", "request is collecting"),
        (bool(state.get("active_request_text")), "active request exists"),
    ]
    if name == "missing_location":
        return common + [
            ("location" in (state.get("missing_slots") or []), "location is pending"),
            (state.get("time_range") is not None, "historical date is retained"),
            ("city or district" in (state.get("clarification_question") or "").lower(), "location was requested"),
        ]
    if name == "independent_request":
        return common + [
            (state.get("active_request_type") == "chart", "old request is a chart"),
            (state.get("chart_type") == "line", "old chart type is line"),
            (state.get("aggregation") == "avg", "old aggregation is average"),
            (any(item.get("value") == "Marconi" for item in state.get("filters", [])), "old location is Marconi"),
        ]
    return common + [
        ("time_range" in (state.get("missing_slots") or []), "time is pending"),
        (any(item.get("value") == "Marconi" for item in state.get("filters", [])), "Marconi is retained"),
        (state.get("aggregation") == "avg", "average aggregation is retained"),
    ]


def passed(checks: list[tuple[bool, str]]) -> bool:
    return all(value for value, _ in checks)


def apply_setup_gate(
    scores: dict[str, Any], *, setup_success: bool, transition_turn: bool
) -> dict[str, Any]:
    gated = deepcopy(scores)
    if transition_turn and not setup_success:
        gated["state_transition_correct"] = False
        gated["functional_completion"] = False
        gated.setdefault("notes", []).append(
            "precondition failed / transition not demonstrated"
        )
    return gated
