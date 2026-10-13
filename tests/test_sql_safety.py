import unittest

from agent.nodes.sql_safety import validate_sql_safety_node
from agent.routers.sql_safety_router import route_sql_safety
from tools.sql_safety_guard_tool import validate_sql_safety_tool


PERMITTED_TABLE = "sensor_measurements"
REQUIREMENTS_CONTEXT = {
    "defaults": {"table": PERMITTED_TABLE},
    "safety": {
        "require_limit": True,
        "max_limit": 5000,
        "allow_only_select": True,
    },
}


class SQLSafetyGuardTests(unittest.TestCase):
    def validate(self, sql: str) -> dict[str, object]:
        return validate_sql_safety_tool(
            sql_query=sql,
            requirements_context=REQUIREMENTS_CONTEXT,
            permitted_table=PERMITTED_TABLE,
        )

    def assert_rejected(self, sql: str, message_fragment: str) -> None:
        result = self.validate(sql)
        self.assertFalse(result["is_safe"], result)
        self.assertTrue(
            any(message_fragment in error for error in result["errors"]),
            result,
        )

    def test_allows_select_from_permitted_table(self) -> None:
        result = self.validate(
            "SELECT timestamp, value FROM sensor_measurements LIMIT 100"
        )
        self.assertEqual(result, {"is_safe": True, "errors": []})

    def test_allows_read_only_cte_backed_by_permitted_table(self) -> None:
        result = self.validate(
            "WITH recent AS ("
            "SELECT timestamp, value FROM sensor_measurements"
            ") SELECT * FROM recent LIMIT 100"
        )
        self.assertEqual(result, {"is_safe": True, "errors": []})

    def test_ignores_forbidden_words_and_semicolons_in_string_literals(self) -> None:
        result = self.validate(
            "SELECT 'drop table; delete rows' AS note "
            "FROM sensor_measurements LIMIT 1"
        )
        self.assertEqual(result, {"is_safe": True, "errors": []})

    def test_comments_do_not_supply_limit_or_sql_operations(self) -> None:
        self.assert_rejected(
            "SELECT value FROM sensor_measurements -- LIMIT 10; DROP TABLE x\n",
            "outermost LIMIT",
        )

        result = self.validate(
            "SELECT value FROM sensor_measurements "
            "/* DROP TABLE x; LIMIT 999999 */ LIMIT 10"
        )
        self.assertEqual(result, {"is_safe": True, "errors": []})

    def test_rejects_missing_and_nested_only_limits(self) -> None:
        cases = (
            "SELECT value FROM sensor_measurements",
            "SELECT 'LIMIT 5' AS text FROM sensor_measurements",
            "WITH limited AS ("
            "SELECT value FROM sensor_measurements LIMIT 5"
            ") SELECT * FROM limited",
            "SELECT * FROM ("
            "SELECT value FROM sensor_measurements LIMIT 5"
            ") AS nested",
        )

        for sql in cases:
            with self.subTest(sql=sql):
                self.assert_rejected(sql, "outermost LIMIT")

    def test_enforces_literal_limit_boundaries(self) -> None:
        for limit in (1, 5000):
            with self.subTest(limit=limit):
                result = self.validate(
                    f"SELECT value FROM sensor_measurements LIMIT {limit}"
                )
                self.assertTrue(result["is_safe"], result)

        for limit in (0, 5001, 999999, -1):
            with self.subTest(limit=limit):
                self.assert_rejected(
                    f"SELECT value FROM sensor_measurements LIMIT {limit}",
                    "Outermost LIMIT",
                )

        for limit_expression in ("@row_limit", "1 + 1", "1.0"):
            with self.subTest(limit_expression=limit_expression):
                self.assert_rejected(
                    "SELECT value FROM sensor_measurements "
                    f"LIMIT {limit_expression}",
                    "literal integer",
                )

    def test_rejects_unrelated_tables_and_unrelated_cte_sources(self) -> None:
        self.assert_rejected(
            "SELECT * FROM unrelated_table LIMIT 10",
            "Unauthorized table reference",
        )
        self.assert_rejected(
            "WITH hidden AS (SELECT * FROM unrelated_table) "
            "SELECT * FROM hidden LIMIT 10",
            "Unauthorized table reference",
        )
        self.assert_rejected(
            "SELECT * FROM `project.dataset.sensor_measurements` LIMIT 10",
            "Qualified physical table",
        )

    def test_rejects_multiple_destructive_and_malformed_sql(self) -> None:
        cases = (
            (
                "SELECT * FROM sensor_measurements LIMIT 1; "
                "SELECT * FROM sensor_measurements LIMIT 1",
                "Exactly one SQL statement",
            ),
            ("DELETE FROM sensor_measurements", "read-only SELECT"),
            ("DROP TABLE sensor_measurements", "read-only SELECT"),
            ("CALL dangerous_procedure()", "read-only SELECT"),
            ("EXECUTE IMMEDIATE 'SELECT 1'", "read-only SELECT"),
            ("SELECT * FROM (", "parsed as valid BigQuery SQL"),
        )

        for sql, message_fragment in cases:
            with self.subTest(sql=sql):
                self.assert_rejected(sql, message_fragment)

    def test_rejects_table_valued_external_and_wildcard_sources(self) -> None:
        cases = (
            (
                "SELECT * FROM EXTERNAL_QUERY('connection', 'SELECT 1') LIMIT 1",
                "Table-valued and external",
            ),
            (
                "SELECT * FROM dataset.table_function(1) LIMIT 1",
                "Table-valued and external",
            ),
            (
                "SELECT * FROM UNNEST([1, 2]) LIMIT 1",
                "Table-valued and external",
            ),
            (
                "SELECT * FROM `project.dataset.events_*` LIMIT 1",
                "Wildcard table",
            ),
        )

        for sql, message_fragment in cases:
            with self.subTest(sql=sql):
                self.assert_rejected(sql, message_fragment)

    def test_router_refuses_execution_after_validation_failure(self) -> None:
        state = {
            "sql_query": "SELECT * FROM unrelated_table LIMIT 10",
            "requirements_context": REQUIREMENTS_CONTEXT,
            "schema_context": {"tables": {PERMITTED_TABLE: {}}},
        }

        validation_update = validate_sql_safety_node(state)
        routed_state = {**state, **validation_update}

        self.assertFalse(validation_update["sql_is_safe"])
        self.assertIn(
            "Unauthorized table reference: unrelated_table.",
            validation_update["sql_validation_errors"],
        )
        self.assertEqual(route_sql_safety(routed_state), "unsafe")


if __name__ == "__main__":
    unittest.main()
