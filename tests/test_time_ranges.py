import unittest

from tools.normalize_time_range_tool import (
    build_bigquery_time_predicate_tool,
    extract_single_explicit_absolute_date_tool,
    normalize_time_range_tool,
)


class TimeRangeRegressionTests(unittest.TestCase):
    def test_single_date_uses_next_midnight_as_exclusive_end(self) -> None:
        time_range = normalize_time_range_tool("on 2026-05-10")

        self.assertEqual(
            time_range,
            {
                "kind": "absolute",
                "relative": None,
                "start": "2026-05-10 00:00:00+00",
                "end": "2026-05-11 00:00:00+00",
            },
        )
        self.assertEqual(
            build_bigquery_time_predicate_tool(time_range, "timestamp"),
            "timestamp >= TIMESTAMP('2026-05-10 00:00:00+00') "
            "AND timestamp < TIMESTAMP('2026-05-11 00:00:00+00')",
        )

    def test_date_range_includes_end_date_via_exclusive_next_day(self) -> None:
        time_range = normalize_time_range_tool(
            "from 2026-05-10 to 2026-05-17"
        )

        self.assertIsNotNone(time_range)
        self.assertEqual(time_range["start"], "2026-05-10 00:00:00+00")
        self.assertEqual(time_range["end"], "2026-05-18 00:00:00+00")

    def test_relative_weeks_are_symbolic_and_compile_to_current_timestamp(self) -> None:
        time_range = normalize_time_range_tool("past 2 weeks")

        self.assertEqual(
            time_range,
            {
                "kind": "relative",
                "amount": 14,
                "unit": "DAY",
                "relative": "last_14_days",
            },
        )
        self.assertEqual(
            build_bigquery_time_predicate_tool(time_range, "timestamp"),
            "timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 14 DAY)",
        )

    def test_single_embedded_absolute_date_is_bounded_and_validated(self) -> None:
        applied = extract_single_explicit_absolute_date_tool(
            "Start a new request: show flow in Ponte on 2026-05-10."
        )
        self.assertEqual(applied["status"], "applied")
        self.assertEqual(applied["phrase"], "2026-05-10")
        self.assertEqual(applied["parsed"]["end"], "2026-05-11 00:00:00+00")

        cases = {
            "Show flow on 2026-02-30.": "invalid",
            "Compare 2026-05-10 with 2026-05-11.": "ambiguous_multiple",
            'Use the example "2026-05-10".': "rejected_quoted",
            "Show flow, not on 2026-05-10.": "rejected_negated",
        }
        for text, expected_status in cases.items():
            with self.subTest(text=text):
                self.assertEqual(
                    extract_single_explicit_absolute_date_tool(text)["status"],
                    expected_status,
                )


if __name__ == "__main__":
    unittest.main()
