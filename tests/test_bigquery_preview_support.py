import unittest
from datetime import datetime, timezone

from google.cloud import bigquery
from google.cloud.bigquery._helpers import _rows_from_json

from scripts.bigquery_preview_support import (
    decode_preview_row_by_name,
    select_preview_fields_in_physical_order,
)


class BigQueryPreviewSupportTests(unittest.TestCase):
    def test_physical_order_drives_sdk_decoding_and_name_access(self) -> None:
        physical_schema = [
            bigquery.SchemaField("sensor_id", "STRING"),
            bigquery.SchemaField("timestamp", "TIMESTAMP"),
            bigquery.SchemaField("type", "STRING"),
            bigquery.SchemaField("value", "FLOAT64"),
            bigquery.SchemaField("city", "STRING"),
            bigquery.SchemaField("district", "STRING"),
        ]
        requested_logical_order = ["district", "timestamp", "value"]
        selected = select_preview_fields_in_physical_order(
            physical_schema,
            requested_logical_order,
        )

        self.assertEqual(
            [field.name for field in selected],
            ["timestamp", "value", "district"],
        )

        synthetic_sdk_response = [
            {
                "f": [
                    {"v": "1689292800000000"},
                    {"v": "12.5"},
                    {"v": "Salza"},
                ]
            }
        ]

        mismatched_schema = [
            next(field for field in physical_schema if field.name == name)
            for name in requested_logical_order
        ]
        with self.assertRaises(ValueError):
            _rows_from_json(synthetic_sdk_response, mismatched_schema)

        decoded_sdk_row = _rows_from_json(synthetic_sdk_response, selected)[0]
        decoded = decode_preview_row_by_name(decoded_sdk_row)

        self.assertEqual(decoded["district"], "Salza")
        self.assertEqual(
            decoded["timestamp"],
            datetime(2023, 7, 14, 0, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(decoded["value"], 12.5)

    def test_missing_required_field_fails_before_preview(self) -> None:
        schema = [
            bigquery.SchemaField("timestamp", "TIMESTAMP"),
            bigquery.SchemaField("value", "FLOAT64"),
        ]

        with self.assertRaisesRegex(ValueError, "missing=district"):
            select_preview_fields_in_physical_order(schema)


if __name__ == "__main__":
    unittest.main()
