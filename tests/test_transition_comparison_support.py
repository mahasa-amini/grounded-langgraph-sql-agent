import unittest
from typing import TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from evaluations.transition_comparison_support import (
    GraphNodeTimingHandler,
    apply_setup_gate,
    build_controlled_fixtures,
    conversation_setup_checks,
    passed,
    validate_controlled_fixture,
)
from agent.utils.evaluation_observability import compact_model_value
from scripts.evaluate_targeted_backend_fixes import selected_check_names
from scripts.evaluate_transition_model_comparison import (
    selected_conversation_scenarios,
    selected_evaluation_tracks,
)
from scripts.evaluate_replacement_paths import (
    NEW_REQUEST,
    assess_scenario_a,
    assess_scenario_b,
)


class TransitionComparisonSupportTests(unittest.TestCase):
    def test_evaluation_compaction_preserves_scalar_types(self) -> None:
        value = {
            "enabled": True,
            "distinct": False,
            "limit": 1000,
            "temperature": 0.0,
            "name": "fixture",
        }

        compacted = compact_model_value(value)

        self.assertIs(compacted["enabled"], True)
        self.assertIs(compacted["distinct"], False)
        self.assertIs(type(compacted["limit"]), int)
        self.assertIs(type(compacted["temperature"]), float)
        self.assertEqual(compacted["name"], "fixture")

    def test_graph_node_timing_handler_records_start_and_completion(self) -> None:
        class TimingState(TypedDict):
            value: int

        events = []
        handler = GraphNodeTimingHandler(events)
        builder = StateGraph(TimingState)
        builder.add_node("increment", lambda state: {"value": state["value"] + 1})
        builder.add_edge(START, "increment")
        builder.add_edge("increment", END)
        result = builder.compile().invoke(
            {"value": 0},
            config={"callbacks": [handler]},
        )

        self.assertEqual(result["value"], 1)
        self.assertEqual(events[0], {"node": "increment", "event": "start"})
        self.assertEqual(events[1]["node"], "increment")
        self.assertEqual(events[1]["event"], "complete")
        self.assertGreaterEqual(events[1]["elapsed_seconds"], 0)

    def test_graph_node_timing_handler_streams_events_to_optional_sink(self) -> None:
        events = []
        streamed = []
        handler = GraphNodeTimingHandler(events, event_sink=streamed.append)

        handler.on_chain_start(
            {"name": "example"},
            {},
            run_id="run-1",
            name="example",
            metadata={"langgraph_node": "example"},
        )
        handler.on_chain_end({}, run_id="run-1")

        self.assertEqual(streamed, events)
        self.assertIsNot(streamed[0], events[0])
        self.assertEqual(streamed[0], {"node": "example", "event": "start"})
        self.assertEqual(streamed[1]["event"], "complete")

    def test_all_controlled_fixtures_are_valid(self) -> None:
        fixtures = build_controlled_fixtures()
        self.assertEqual(
            {name: validate_controlled_fixture(name, fixture) for name, fixture in fixtures.items()},
            {name: [] for name in fixtures},
        )

    def test_independent_setup_requires_real_old_request_fields(self) -> None:
        invalid = {
            "request_status": "idle",
            "active_request_text": None,
            "active_request_type": None,
            "chart_type": None,
            "aggregation": None,
            "filters": [],
        }
        self.assertFalse(passed(conversation_setup_checks("independent_request", invalid)))

        valid = build_controlled_fixtures()["independent_request"]["state"]
        self.assertTrue(passed(conversation_setup_checks("independent_request", valid)))

    def test_failed_setup_cannot_score_transition_or_functional_pass(self) -> None:
        result = apply_setup_gate(
            {
                "classification_compliance": True,
                "state_transition_correct": True,
                "functional_completion": True,
                "notes": [],
            },
            setup_success=False,
            transition_turn=True,
        )
        self.assertFalse(result["state_transition_correct"])
        self.assertFalse(result["functional_completion"])
        self.assertIn(
            "precondition failed / transition not demonstrated",
            result["notes"],
        )

    def test_targeted_runner_can_select_only_the_pending_date_check(self) -> None:
        self.assertEqual(
            selected_check_names("pending_time_bare_date"),
            ("pending_time_bare_date",),
        )
        with self.assertRaises(ValueError):
            selected_check_names("not-a-check")

    def test_comparison_runner_can_select_only_conversation_track(self) -> None:
        self.assertEqual(
            selected_evaluation_tracks("conversation"),
            ("conversation",),
        )
        with self.assertRaises(ValueError):
            selected_evaluation_tracks("not-a-track")

    def test_comparison_runner_can_select_exact_conversation_scenarios(self) -> None:
        scenarios = selected_conversation_scenarios(
            "missing_location,independent_request"
        )
        self.assertEqual(
            [name for name, _ in scenarios],
            ["missing_location", "independent_request"],
        )
        with self.assertRaises(ValueError):
            selected_conversation_scenarios("not-a-scenario")
        with self.assertRaises(ValueError):
            selected_conversation_scenarios("missing_location,missing_location")

    def test_supplementary_replacement_assessments_require_fresh_state_and_safe_sql(self) -> None:
        setup = [{"passed": True, "description": "fixture"}]
        lifecycle = {
            "active_request_text": NEW_REQUEST,
            "active_request_type": "data",
            "output_type": "text",
            "chart_type": None,
            "aggregation": None,
            "filters": [],
            "clarification_answers": [],
            "clarification_count": 0,
            "pending_ambiguous_message": None,
            "sql_query": None,
            "query_result": None,
        }
        sql_after = {
            "sql_query": "SELECT value FROM sensor_measurements WHERE district = 'Ponte' AND timestamp >= '2026-05-10' AND timestamp < '2026-05-11' LIMIT 1000",
            "sql_is_safe": True,
            "query_result": {"execution_label": "EVALUATION_STUB_NO_DATABASE_EXECUTION"},
            "messages": [],
        }
        turns = [
            {"turn": 1, "setup_checks": setup, "classification": {}},
            {
                "turn": 2,
                "classification": {},
                "after_full": {
                    "active_request_text": "Show a line chart of average flow in Marconi.",
                    "chart_type": "line", "aggregation": "avg",
                    "filters": [{"value": "Marconi"}],
                    "pending_ambiguous_message": NEW_REQUEST,
                    "messages": [HumanMessage(content="Show a line chart of average flow in Marconi."), HumanMessage(content=NEW_REQUEST)],
                },
            },
            {"turn": 3, "classification": {}, "lifecycle": lifecycle, "mapping": {"filters": [{"value": "Ponte"}]}},
            {"turn": 4, "classification": {}, "lifecycle": {"active_request_text": NEW_REQUEST, "clarification_answers": ["2026-05-10"]}, "after_full": sql_after, "execution_label": "EVALUATION_STUB_NO_DATABASE_EXECUTION"},
        ]
        assessment = assess_scenario_a(turns, "relationship_clarification")
        self.assertFalse(assessment["passed"])
        self.assertFalse(assessment["completion"]["conversation_history_preserved"])

        explicit = assess_scenario_b([
            {"turn": 1, "setup_checks": setup},
            {"turn": 2, "classification": {"active_request_action": "new_request", "model_classification": {"active_request_action": "new_request"}}, "lifecycle": lifecycle, "mapping": {"filters": [{"value": "Ponte"}]}, "after_full": sql_after, "execution_label": "EVALUATION_STUB_NO_DATABASE_EXECUTION"},
        ])
        self.assertFalse(explicit["passed"])
        self.assertFalse(explicit["fresh_initialization"]["conversation_history_preserved"])


if __name__ == "__main__":
    unittest.main()
