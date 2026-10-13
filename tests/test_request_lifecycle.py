import unittest
from unittest.mock import Mock, patch

from agent.nodes.clarification import (
    answer_clarification_node,
    check_clarification_limit_node,
)
from agent.nodes.completeness import check_completeness_node
from agent.nodes.mapping_validation import validate_mapped_state_node
from agent.nodes.request_evidence import extract_request_evidence_node
from agent.nodes.request_interpretation import interpret_user_request_node
from agent.nodes.request_lifecycle import (
    mark_request_done_node,
    reset_active_request_node,
    start_or_update_active_request_node,
)
from agent.nodes.request_validation import pre_validate_request_evidence_node
from agent.nodes.schema_mapping import map_to_schema_node
from agent.nodes.time_parsing import parse_time_phrase_node
from agent.routers.clarification_router import route_clarification_limit
from tools.schema_context_tool import get_schema_context_tool


class RequestLifecycleRegressionTests(unittest.TestCase):
    def test_new_request_replaces_stale_completed_request_state(self) -> None:
        update = start_or_update_active_request_node(
            {
                "latest_user_message": "show flow in Marconi yesterday",
                "user_message_type": "data_request",
                "active_request_action": "new_request",
                "classification_valid": True,
                "request_status": "idle",
                "sql_query": "SELECT stale",
                "query_result": {"rows": [{"value": 99}]},
                "clarification_answers": ["stale answer"],
            }
        )

        self.assertEqual(update["request_status"], "collecting")
        self.assertEqual(update["active_request_type"], "data")
        self.assertEqual(update["output_type"], "text")
        self.assertEqual(
            update["active_request_text"],
            "show flow in Marconi yesterday",
        )
        self.assertEqual(update["clarification_answers"], [])
        self.assertIsNone(update["sql_query"])
        self.assertIsNone(update["query_result"])

    def test_completed_request_clears_working_state_but_keeps_result(self) -> None:
        query_result = {"success": True, "rows": [{"value": 1.25}], "row_count": 1}
        update = mark_request_done_node(
            {
                "final_answer": "done",
                "sql_query": "SELECT value FROM sensor_measurements LIMIT 1",
                "query_result": query_result,
                "request_status": "ready_for_sql",
                "active_request_type": "data",
                "active_request_text": "show flow",
                "clarification_question": "which date?",
                "clarification_answers": ["yesterday"],
                "clarification_count": 1,
            }
        )

        self.assertEqual(update["request_status"], "idle")
        self.assertIsNone(update["active_request_text"])
        self.assertIsNone(update["clarification_question"])
        self.assertEqual(update["clarification_answers"], [])
        self.assertEqual(update["clarification_count"], 0)
        self.assertNotIn("sql_query", update)
        self.assertNotIn("query_result", update)

    def test_clarification_increments_count_and_hits_configured_limit(self) -> None:
        state = {
            "missing_slots": ["time_range"],
            "clarification_count": 1,
            "max_clarification_count": 2,
            "request_status": "collecting",
        }

        update = answer_clarification_node(state)
        routed_state = {**state, **update}

        self.assertEqual(update["request_status"], "collecting")
        self.assertEqual(update["clarification_count"], 2)
        self.assertEqual(route_clarification_limit(routed_state), "limit_exceeded")
        self.assertEqual(
            check_clarification_limit_node(routed_state)["max_clarification_count"],
            2,
        )

    def test_reset_clears_request_and_clarification_state(self) -> None:
        state = {
            "request_status": "collecting",
            "active_request_type": "chart",
            "active_request_text": "chart flow",
            "clarification_question": "which date?",
            "clarification_answers": ["yesterday"],
            "clarification_count": 3,
        }
        update = reset_active_request_node(state)
        next_state = {**state, **update}

        self.assertEqual(next_state["request_status"], "idle")
        self.assertIsNone(next_state["active_request_text"])
        self.assertEqual(next_state["clarification_answers"], [])
        self.assertEqual(
            (
                next_state["clarification_question"],
                next_state["clarification_count"],
            ),
            (None, 0),
        )

    def test_date_clarification_preserves_collected_request_state(self) -> None:
        original_request = "Show average flow in Marconi."
        clarification_answer = "2026-05-10"
        schema_data = get_schema_context_tool()

        state = {
            **schema_data,
            "latest_user_message": clarification_answer,
            "user_message_type": "clarification_or_update",
            "active_request_action": "clarification_answer",
            "classification_valid": True,
            "request_status": "collecting",
            "active_request_type": "data",
            "output_type": "text",
            "active_request_text": original_request,
            "clarification_question": "For which date?",
            "clarification_answers": [],
            "clarification_count": 1,
            "missing_slots": ["time_range"],
            "evidence_suggestions": {},
            "table_name": "sensor_measurements",
            "available_columns": [
                "sensor_id",
                "timestamp",
                "type",
                "value",
                "city",
                "district",
            ],
            "intent": "retrieve_data",
            "raw_metric": "flow",
            "metric_column": "value",
            "time_column": "timestamp",
            "time_range": None,
            "filters": [
                {"column": "district", "operator": "=", "value": "Marconi"}
            ],
            "selected_columns": [],
            "aggregation": "avg",
            "group_by": [],
            "order_by": None,
            "limit": 1000,
            "unsupported_terms": [],
            "ambiguous_terms": [],
        }

        lifecycle_update = start_or_update_active_request_node(state)
        state = {**state, **lifecycle_update}

        self.assertEqual(state["active_request_text"], original_request)
        self.assertEqual(state["clarification_answers"], [clarification_answer])
        self.assertEqual(
            state["mapping_input_text"],
            f"{original_request}\n{clarification_answer}",
        )
        self.assertEqual(state["active_request_type"], "data")

        state = {**state, **interpret_user_request_node(state)}

        evidence_chain = Mock()
        evidence_chain.invoke.return_value = {
            "metric_phrase": "flow",
            "location_phrases": ["Marconi"],
            "time_phrase": clarification_answer,
            "aggregation_phrase": "average",
            "chart_phrase": None,
        }

        with patch(
            "agent.nodes.request_evidence.get_request_evidence_chain",
            return_value=evidence_chain,
        ):
            state = {**state, **extract_request_evidence_node(state)}

        evidence_input = evidence_chain.invoke.call_args.args[0]
        self.assertEqual(
            evidence_input["mapping_input_text"],
            f"{original_request}\n{clarification_answer}",
        )

        state = {**state, **parse_time_phrase_node(state)}
        state = {**state, **pre_validate_request_evidence_node(state)}

        mapping_chain = Mock()
        mapping_chain.invoke.return_value = {
            "table_name": None,
            "active_request_type": None,
            "output_type": None,
            "chart_type": None,
            "metric_column": None,
            "raw_metric": None,
            "time_column": None,
            "time_range": None,
            "filters": [],
            "selected_columns": [],
            "aggregation": None,
            "group_by": [],
            "order_by": None,
            "limit": None,
            "missing_slots": [],
            "unsupported_terms": [],
            "ambiguous_terms": [],
        }

        with patch(
            "agent.nodes.schema_mapping.get_schema_mapping_chain",
            return_value=mapping_chain,
        ):
            state = {**state, **map_to_schema_node(state)}

        state = {**state, **validate_mapped_state_node(state)}
        state = {**state, **check_completeness_node(state)}

        self.assertEqual(state["active_request_text"], original_request)
        self.assertEqual(
            state["mapping_input_text"],
            f"{original_request}\n{clarification_answer}",
        )
        self.assertEqual(state["active_request_type"], "data")
        self.assertEqual(state["output_type"], "text")
        self.assertEqual(state["metric_column"], "value")
        self.assertEqual(state["aggregation"], "avg")
        self.assertIn(
            {"column": "district", "operator": "=", "value": "Marconi"},
            state["filters"],
        )
        self.assertEqual(
            state["time_range"],
            {
                "kind": "absolute",
                "relative": None,
                "start": "2026-05-10 00:00:00+00",
                "end": "2026-05-11 00:00:00+00",
            },
        )
        self.assertTrue(state["is_complete"])
        self.assertEqual(state["request_status"], "ready_for_sql")


if __name__ == "__main__":
    unittest.main()
