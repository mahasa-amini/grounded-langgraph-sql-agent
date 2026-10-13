import unittest
from unittest.mock import Mock, patch

from agent.nodes.sql_generation import generate_sql_node
from agent.nodes.sql_safety import validate_sql_safety_node
from agent.routers.sql_safety_router import route_sql_safety
from agent.utils.evaluation_observability import capture_evaluation_observability
from tools.schema_context_tool import get_schema_context_tool
from tools.sql_compiler_tool import compile_validated_sql_plan_tool


class DeterministicSQLCompilerTests(unittest.TestCase):
    def base_plan(self) -> dict:
        contexts = get_schema_context_tool()
        return {
            **contexts,
            "table_name": "sensor_measurements",
            "available_columns": [
                "sensor_id", "timestamp", "type", "value", "city", "district"
            ],
            "selected_columns": ["timestamp", "value"],
            "metric_column": "value",
            "time_column": "timestamp",
            "time_range": {
                "kind": "absolute",
                "start": "2026-05-10 00:00:00+00",
                "end": "2026-05-11 00:00:00+00",
                "relative": None,
            },
            "filters": [
                {"column": "district", "operator": "=", "value": "Ponte"}
            ],
            "aggregation": None,
            "group_by": [],
            "order_by": "timestamp ASC",
            "having_conditions": [],
            "distinct": False,
            "limit": 1000,
            "unsupported_terms": [],
            "ambiguous_terms": [],
        }

    def compile(self, **overrides):
        return compile_validated_sql_plan_tool({**self.base_plan(), **overrides})

    def assert_safe(self, result: dict) -> None:
        self.assertEqual(result["errors"], [])
        state = {**self.base_plan(), "sql_query": result["sql_query"]}
        safety = validate_sql_safety_node(state)
        self.assertTrue(safety["sql_is_safe"], safety["sql_validation_errors"])

    def test_raw_ponte_interval_compiles_with_exclusive_end_once(self) -> None:
        result = self.compile()

        self.assert_safe(result)
        sql = result["sql_query"]
        self.assertIn("SELECT timestamp, value FROM sensor_measurements", sql)
        self.assertIn("district = 'Ponte'", sql)
        self.assertIn("timestamp >= TIMESTAMP('2026-05-10 00:00:00+00')", sql)
        self.assertIn("timestamp < TIMESTAMP('2026-05-11 00:00:00+00')", sql)
        self.assertEqual(sql.count("timestamp >="), 1)
        self.assertEqual(sql.count("LIMIT 1000"), 1)
        self.assertNotIn("DISTINCT", sql)

    def test_scalar_average_does_not_project_timestamp_or_raw_value(self) -> None:
        result = self.compile(
            selected_columns=[], aggregation="avg", order_by=None
        )

        self.assert_safe(result)
        projection = result["sql_query"].split(" FROM ", 1)[0]
        self.assertEqual(projection, "SELECT AVG(value) AS avg_value")

    def test_grouped_chart_shape_having_ordering_and_limit(self) -> None:
        result = self.compile(
            selected_columns=["district"],
            filters=[{"column": "city", "operator": "=", "value": "Marene"}],
            aggregation="avg",
            group_by=["district"],
            order_by="avg_value DESC",
            having_conditions=[{"column": "value", "operator": ">", "value": 2}],
            limit=25,
        )

        self.assert_safe(result)
        sql = result["sql_query"]
        self.assertIn("SELECT district, AVG(value) AS avg_value", sql)
        self.assertIn("GROUP BY district", sql)
        self.assertIn("HAVING AVG(value) > 2", sql)
        self.assertIn("ORDER BY avg_value DESC", sql)
        self.assertTrue(sql.endswith("LIMIT 25"))

    def test_distinct_and_typed_literals_are_compiled_without_raw_interpolation(self) -> None:
        result = self.compile(
            selected_columns=["sensor_id"],
            filters=[
                {"column": "sensor_id", "operator": "=", "value": "O'Reilly"},
                {"column": "value", "operator": ">=", "value": 1.5},
            ],
            distinct=True,
            order_by=None,
        )

        self.assert_safe(result)
        sql = result["sql_query"]
        self.assertIn("SELECT DISTINCT sensor_id", sql)
        self.assertIn("sensor_id = 'O\\'Reilly'", sql)
        self.assertIn("value >= 1.5", sql)

    def test_in_and_between_require_typed_nonempty_values(self) -> None:
        valid = self.compile(
            filters=[
                {"column": "district", "operator": "IN", "value": ["Ponte", "Marconi"]},
                {"column": "value", "operator": "BETWEEN", "value": [1, 3.5]},
            ]
        )
        self.assert_safe(valid)
        self.assertIn("district IN ('Ponte', 'Marconi')", valid["sql_query"])
        self.assertIn("value BETWEEN 1 AND 3.5", valid["sql_query"])

        invalid = self.compile(
            filters=[{"column": "district", "operator": "IN", "value": "Ponte"}]
        )
        self.assertIsNone(invalid["sql_query"])
        self.assertTrue(any("non-empty list" in error for error in invalid["errors"]))

    def test_current_contract_rejects_time_bucket_instead_of_omitting_it(self) -> None:
        result = self.compile(time_bucket="hour")

        self.assertIsNone(result["sql_query"])
        self.assertTrue(any("time_bucket" in error for error in result["errors"]))

    def test_invalid_identifiers_types_operators_and_incomplete_plans_fail(self) -> None:
        cases = [
            {"table_name": "sensor_measurements; DROP TABLE x"},
            {"selected_columns": ["value FROM secret"]},
            {"filters": [{"column": "district", "operator": "LIKE", "value": "%"}]},
            {"filters": [{"column": "value", "operator": ">", "value": "one"}]},
            {"selected_columns": []},
            {"limit": "1000"},
            {"time_range": {"kind": "absolute", "start": "bad", "end": None}},
        ]
        for overrides in cases:
            with self.subTest(overrides=overrides):
                result = self.compile(**overrides)
                self.assertIsNone(result["sql_query"])
                self.assertTrue(result["errors"])

    def test_conflicting_aggregate_projection_group_filter_and_order_fail(self) -> None:
        cases = [
            {"aggregation": "avg", "selected_columns": ["timestamp"], "order_by": None},
            {"aggregation": "avg", "selected_columns": [], "distinct": True, "order_by": None},
            {
                "aggregation": "avg",
                "selected_columns": ["district"],
                "group_by": ["district"],
                "order_by": None,
            },
            {"aggregation": "avg", "selected_columns": [], "order_by": "timestamp ASC"},
            {"aggregation": None, "having_conditions": [{"column": "value", "operator": ">", "value": 1}]},
        ]
        for overrides in cases:
            with self.subTest(overrides=overrides):
                result = self.compile(**overrides)
                self.assertIsNone(result["sql_query"])
                self.assertTrue(result["errors"])

    def test_supported_normal_path_never_calls_sql_model(self) -> None:
        forbidden_chain = Mock(side_effect=AssertionError("SQL model must not be constructed"))
        with (
            patch(
                "agent.nodes.sql_generation.get_sql_generation_chain",
                forbidden_chain,
            ),
            capture_evaluation_observability() as events,
        ):
            update = generate_sql_node(self.base_plan())

        forbidden_chain.assert_not_called()
        self.assertEqual(update["sql_generation_provenance"], "deterministic_compiler")
        self.assertEqual(update["sql_compilation_errors"], [])
        boundary = next(event for event in events if event["stage"] == "sql_generation_boundary")
        self.assertFalse(boundary["sql_model_call"])
        self.assertIn("not applicable", boundary["raw_provider_response"])

    def test_rejected_plan_routes_away_from_execution(self) -> None:
        generated = generate_sql_node({**self.base_plan(), "limit": "1000"})
        state = {**self.base_plan(), **generated}
        safety = validate_sql_safety_node(state)

        self.assertIsNone(generated["sql_query"])
        self.assertFalse(safety["sql_is_safe"])
        self.assertEqual(route_sql_safety({**state, **safety}), "unsafe")


if __name__ == "__main__":
    unittest.main()
