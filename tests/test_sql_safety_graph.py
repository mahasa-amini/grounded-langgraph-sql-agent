import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import HumanMessage

from agent.graph import build_graph


class SQLSafetyGraphRegressionTests(unittest.TestCase):
    def test_rejected_sql_does_not_reach_execution_node(self) -> None:
        execute_sql = Mock(side_effect=AssertionError("execution must not run"))

        def classify_data_request(_state):
            return {
                "user_message_type": "data_request",
                "classification_reason": "offline regression fixture",
                "active_request_action": "new_request",
                "classification_valid": True,
                "classified_request_type": "data",
            }

        def extract_request_evidence(_state):
            return {
                "mapping_input_text": "show flow in Marconi on 2026-05-10",
                "request_evidence": {
                    "metric_phrase": "flow",
                    "location_phrases": ["Marconi"],
                    "time_phrase": "on 2026-05-10",
                    "aggregation_phrase": None,
                    "chart_phrase": None,
                },
            }

        def map_supported_request(state):
            return {
                "table_name": "sensor_measurements",
                "raw_metric": "flow",
                "metric_column": "value",
                "time_column": "timestamp",
                "time_range": state.get("parsed_time_range"),
                "filters": [
                    {"column": "district", "operator": "=", "value": "Marconi"}
                ],
                "selected_columns": ["timestamp", "value"],
                "aggregation": None,
                "group_by": [],
                "order_by": "timestamp ASC",
                "limit": 100,
                "unsupported_terms": [],
                "ambiguous_terms": [],
            }

        def generate_unauthorized_sql(_state):
            return {
                "sql_query": "SELECT value FROM unrelated_table LIMIT 10",
                "sql_explanation": "offline unsafe fixture",
            }

        with (
            patch(
                "agent.graph.classify_user_message_node",
                new=classify_data_request,
            ),
            patch(
                "agent.graph.extract_request_evidence_node",
                new=extract_request_evidence,
            ),
            patch("agent.graph.map_to_schema_node", new=map_supported_request),
            patch("agent.graph.generate_sql_node", new=generate_unauthorized_sql),
            patch("agent.graph.execute_sql_node", new=execute_sql),
        ):
            graph = build_graph(use_checkpointer=False)
            result = graph.invoke(
                {
                    "messages": [
                        HumanMessage(
                            content="show flow in Marconi on 2026-05-10"
                        )
                    ]
                }
            )

        execute_sql.assert_not_called()
        self.assertFalse(result["sql_is_safe"])
        self.assertIn(
            "Unauthorized table reference: unrelated_table.",
            result["sql_validation_errors"],
        )
        self.assertIn("read-only SELECT queries", result["final_answer"])


if __name__ == "__main__":
    unittest.main()
