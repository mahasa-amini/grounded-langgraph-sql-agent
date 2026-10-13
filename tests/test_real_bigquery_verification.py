from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import unittest
from unittest.mock import patch

from google.cloud import bigquery
from google.cloud.bigquery._helpers import _rows_from_json

from agent.graph import build_graph
from scripts.real_bigquery_verification import (
    MAXIMUM_BYTES_BILLED,
    QueryLedger,
    VerificationDependencies,
    VerificationError,
    run_verification,
)
from tools.schema_context_tool import get_schema_context_tool


@dataclass
class FakeConfig:
    gcp_project_id: str = "synthetic-project"
    bigquery_dataset_id: str = "synthetic_dataset"
    bigquery_table_id: str = "synthetic_table"
    bigquery_location: str = "EU"


class FakeJob:
    def __init__(self, rows=(), processed=4096):
        self._rows = list(rows)
        self.total_bytes_processed = processed
        self.job_id = "synthetic-job"

    def result(self, *args, **kwargs):
        return list(self._rows)


class FakeClient:
    def __init__(self, preview_rows, *, dry_bytes=4096):
        self.preview_rows = preview_rows
        self.dry_bytes = dry_bytes
        self.metadata_calls = 0
        self.preview_calls = 0
        self.query_calls = []
        self.table = bigquery.Table("synthetic-project.synthetic_dataset.synthetic_table")
        self.table.schema = [
            bigquery.SchemaField("sensor_id", "STRING"),
            bigquery.SchemaField("timestamp", "TIMESTAMP"),
            bigquery.SchemaField("type", "STRING"),
            bigquery.SchemaField("value", "FLOAT64"),
            bigquery.SchemaField("city", "STRING"),
            bigquery.SchemaField("district", "STRING"),
        ]
        self.table._properties["location"] = "EU"

    def get_table(self, *args, **kwargs):
        self.metadata_calls += 1
        return self.table

    def list_rows(self, table, *, selected_fields, max_results, retry, timeout):
        self.preview_calls += 1
        if [field.name for field in selected_fields] != ["timestamp", "value", "district"]:
            raise AssertionError("preview fields did not preserve physical schema order")
        if max_results != 100 or retry is not None:
            raise AssertionError("preview boundary was not bounded")
        return self.preview_rows

    def query(self, sql, *, job_config, location, retry, timeout):
        self.query_calls.append((sql, job_config.dry_run, retry, timeout, job_config))
        if retry is not None or job_config.maximum_bytes_billed != MAXIMUM_BYTES_BILLED:
            raise AssertionError("query retry/cap boundary was not enforced")
        if job_config.destination is not None:
            raise AssertionError("permanent destination is forbidden")
        if job_config.dry_run:
            return FakeJob(processed=self.dry_bytes)
        if "reference_average" in sql:
            return FakeJob([{"reference_average": 1.0}])
        if "AVG(" in sql.upper():
            return FakeJob([{"avg_value": 1.0}])
        rows = [
            {"timestamp": datetime(2023, 7, 14, 0, 0, tzinfo=timezone.utc), "value": 0.0},
            {"timestamp": datetime(2023, 7, 14, 0, 15, tzinfo=timezone.utc), "value": 2.0},
        ]
        return FakeJob(rows)


class FakeChain:
    def __init__(self, kind):
        self.kind = kind

    def invoke(self, payload):
        if self.kind == "classification":
            return {
                "message_type": "data_request",
                "reason": "synthetic full-runner fixture",
                "confidence": 1.0,
                "active_request_action": "new_request",
                "correction_targets": [],
                "resolved_request_type": "data",
            }
        text = payload.get("mapping_input_text", "")
        aggregate = "average" in text.lower()
        if self.kind == "evidence":
            return {
                "metric_phrase": "water flow",
                "location_phrases": ["Salza"],
                "time_phrase": "2023-07-14",
                "aggregation_phrase": "average" if aggregate else None,
                "chart_phrase": None,
            }
        return {
            "table_name": "sensor_measurements",
            "active_request_type": "data",
            "output_type": "text",
            "chart_type": None,
            "metric_column": "value",
            "raw_metric": "flow",
            "time_column": "timestamp",
            "time_range": None,
            "filters": [{"column": "district", "operator": "=", "value": "Salza"}],
            "selected_columns": [] if aggregate else ["timestamp", "value"],
            "aggregation": "avg" if aggregate else None,
            "group_by": [],
            "order_by": None if aggregate else "timestamp ASC",
            "limit": 1 if aggregate else 100,
            "missing_slots": [],
            "unsupported_terms": [],
            "ambiguous_terms": [],
        }


def synthetic_preview(*, value="0"):
    schema = [
        bigquery.SchemaField("timestamp", "TIMESTAMP"),
        bigquery.SchemaField("value", "FLOAT64"),
        bigquery.SchemaField("district", "STRING"),
    ]
    return _rows_from_json(
        [{"f": [{"v": "1689292800000000"}, {"v": value}, {"v": "Salza"}]}],
        schema,
    )


class RealBigQueryVerificationRunnerTests(unittest.TestCase):
    def _dependencies(self, client, loader=get_schema_context_tool):
        return VerificationDependencies(
            schema_loader=loader,
            graph_factory=lambda: build_graph(use_checkpointer=False),
            client_factory=lambda **kwargs: client,
        )

    def _run(self, client, loader=get_schema_context_tool):
        with (
            patch("agent.nodes.message_classification.get_message_classification_chain", return_value=FakeChain("classification")),
            patch("agent.nodes.request_evidence.get_request_evidence_chain", return_value=FakeChain("evidence")),
            patch("agent.nodes.schema_mapping.get_schema_mapping_chain", return_value=FakeChain("mapping")),
        ):
            return run_verification(
                config=FakeConfig(),
                dependencies=self._dependencies(client, loader),
                tested_commit="offline-tested-commit",
                attempt_label="offline-full-workflow",
            )

    def test_complete_workflow_uses_actual_loader_graph_and_typed_zero_preview(self):
        client = FakeClient(synthetic_preview(value="0"))
        report = self._run(client)
        self.assertEqual(report["preview_calls"], 1)
        self.assertEqual(report["preview_rows_scanned"], 1)
        self.assertEqual(report["model_calls"], 6)
        self.assertEqual(report["backend_live_selects"], 2)
        self.assertEqual(report["reference_live_selects"], 2)
        self.assertEqual(report["raw_comparison"], "PASS")
        self.assertEqual(report["average_comparison"], "PASS")
        self.assertTrue(report["zero_database_modifications"])
        self.assertEqual(len(client.query_calls), 8)

    def test_recorded_wrong_public_schema_key_stops_before_preview(self):
        client = FakeClient(synthetic_preview())
        payload = get_schema_context_tool()
        payload["public_schema"] = payload.pop("public_schema_context")
        with self.assertRaisesRegex(VerificationError, "public_schema_context"):
            self._run(client, loader=lambda: payload)
        self.assertEqual(client.metadata_calls, 0)
        self.assertEqual(client.preview_calls, 0)
        self.assertEqual(client.query_calls, [])

    def test_empty_preview_stops_without_model_or_exploratory_query(self):
        client = FakeClient([])
        with self.assertRaisesRegex(VerificationError, "bounded preview"):
            self._run(client)
        self.assertEqual(client.preview_calls, 1)
        self.assertEqual(client.query_calls, [])

    def test_null_only_preview_stops_without_query(self):
        client = FakeClient(synthetic_preview(value=None))
        with self.assertRaisesRegex(VerificationError, "bounded preview"):
            self._run(client)
        self.assertEqual(client.preview_calls, 1)
        self.assertEqual(client.query_calls, [])

    def test_unsupported_preview_selection_stops_without_query(self):
        rows = synthetic_preview()
        # Name-based immutable Row access is retained; build a fresh unsupported row.
        schema = [bigquery.SchemaField("timestamp", "TIMESTAMP"), bigquery.SchemaField("value", "FLOAT64"), bigquery.SchemaField("district", "STRING")]
        rows = _rows_from_json([{"f": [{"v": "1689292800000000"}, {"v": "1"}, {"v": "Unsupported"}]}], schema)
        client = FakeClient(rows)
        with self.assertRaisesRegex(VerificationError, "bounded preview"):
            self._run(client)
        self.assertEqual(client.query_calls, [])

    def test_dry_run_cap_stops_before_any_live_query_or_second_request(self):
        client = FakeClient(synthetic_preview(), dry_bytes=MAXIMUM_BYTES_BILLED + 1)
        with self.assertRaisesRegex(VerificationError, "raw graph execution failed"):
            self._run(client)
        self.assertEqual(len(client.query_calls), 1)
        self.assertTrue(client.query_calls[0][1])

    def test_live_call_budget_rejects_third_select(self):
        ledger = QueryLedger()
        ledger.reserve("backend", dry_run=False)
        ledger.reserve("backend", dry_run=False)
        with self.assertRaisesRegex(VerificationError, "budget exhausted"):
            ledger.reserve("backend", dry_run=False)

    def test_public_report_has_no_rows_identifiers_or_measurement_values(self):
        report = self._run(FakeClient(synthetic_preview()))
        rendered = json.dumps(report)
        for forbidden in ("synthetic-project", "synthetic_dataset", "synthetic_table", "Salza", "2023-07-14T00:00", '"value"'):
            self.assertNotIn(forbidden, rendered)


if __name__ == "__main__":
    unittest.main()
