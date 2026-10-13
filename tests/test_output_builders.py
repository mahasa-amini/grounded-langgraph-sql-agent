import unittest

from agent.nodes.chart_builder import build_chart_payload_node
from agent.nodes.response_builder import build_text_response_node


SQL = "SELECT timestamp, value FROM sensor_measurements LIMIT 10"


class OutputBuilderRegressionTests(unittest.TestCase):
    def test_text_output_handles_empty_result(self) -> None:
        update = build_text_response_node(
            {
                "query_result": {"success": True, "rows": [], "row_count": 0},
                "execution_error": None,
                "sql_query": SQL,
                "output_type": "text",
            }
        )

        self.assertIn("no rows matched", update["final_answer"])
        self.assertIn(SQL, update["final_answer"])
        self.assertEqual(update["messages"][0].content, update["final_answer"])

    def test_text_output_formats_representative_rows_safely(self) -> None:
        update = build_text_response_node(
            {
                "query_result": {
                    "success": True,
                    "rows": [
                        {"district": "Marconi|North", "avg_value": 1.23456789},
                        {"district": "Ponte\nSouth", "avg_value": None},
                    ],
                    "row_count": 2,
                },
                "execution_error": None,
                "sql_query": SQL,
                "output_type": "text",
            }
        )

        answer = update["final_answer"]
        self.assertIn("returned 2 row(s)", answer)
        self.assertIn("Marconi\\|North", answer)
        self.assertIn("Ponte South", answer)
        self.assertIn("1.23457", answer)

    def test_chart_output_handles_empty_result_without_payload(self) -> None:
        update = build_chart_payload_node(
            {
                "query_result": {"success": True, "rows": [], "row_count": 0},
                "execution_error": None,
                "sql_query": SQL,
                "output_type": "chart",
                "chart_type": "line",
            }
        )

        self.assertIsNone(update["chart_payload"])
        self.assertIn("no data to chart", update["final_answer"])
        self.assertIn(SQL, update["final_answer"])

    def test_chart_output_builds_line_payload_from_fixture(self) -> None:
        rows = [
            {"timestamp": "2026-05-10T00:00:00+00:00", "value": 1.5},
            {"timestamp": "2026-05-10T01:00:00+00:00", "value": 1.75},
        ]
        update = build_chart_payload_node(
            {
                "query_result": {"success": True, "rows": rows, "row_count": 2},
                "execution_error": None,
                "sql_query": SQL,
                "output_type": "chart",
                "chart_type": "line",
                "time_column": "timestamp",
                "metric_column": "value",
                "aggregation": None,
                "group_by": [],
            }
        )

        self.assertEqual(
            update["chart_payload"],
            {
                "type": "line",
                "title": "Value by Timestamp (line)",
                "x_key": "timestamp",
                "x_label": "Timestamp",
                "y_key": "value",
                "y_label": "Value",
                "series": [{"data_key": "value", "label": "Value"}],
                "data": rows,
            },
        )
        self.assertIn("prepared a line chart payload", update["final_answer"])


if __name__ == "__main__":
    unittest.main()
