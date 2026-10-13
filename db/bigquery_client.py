from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
import re
from typing import Any

from google.api_core.exceptions import GoogleAPIError
from google.cloud import bigquery

from app.config import settings


def build_bigquery_physical_table_name() -> str:
    """Return the configured fully-qualified BigQuery table name."""
    return settings.bigquery_physical_table_name


def replace_logical_table_with_physical_table(
    sql_query: str,
    logical_table_name: str | None = None,
    physical_table_name: str | None = None,
) -> str:
    """Replace logical table references with the configured physical BigQuery table."""
    if not sql_query:
        return sql_query

    logical_name = logical_table_name or settings.bigquery_logical_table_name
    physical_name = physical_table_name or build_bigquery_physical_table_name()

    escaped_logical_name = re.escape(logical_name)

    table_reference_pattern = rf"(?:`{escaped_logical_name}`|{escaped_logical_name})"
    pattern = rf"(?P<prefix>\b(?:FROM|JOIN)\s+){table_reference_pattern}(?P<after>[\s,]|$)"

    return re.sub(
        pattern=pattern,
        repl=lambda match: (
            f"{match.group('prefix')}"
            f"{physical_name}"
            f"{match.group('after')}"
        ),
        string=sql_query,
        flags=re.IGNORECASE,
    )


def prepare_sql_for_bigquery_execution(sql_query: str) -> str:
    """Prepare generated SQL for BigQuery execution."""
    return replace_logical_table_with_physical_table(sql_query=sql_query)


def execute_bigquery_select(sql_query: str) -> dict[str, Any]:
    """Execute a read-only SELECT query in BigQuery and return normalized rows."""
    executable_sql = prepare_sql_for_bigquery_execution(sql_query)

    try:
        client = bigquery.Client(
            project=settings.gcp_project_id,
            location=settings.bigquery_location,
        )

        query_job_config = bigquery.QueryJobConfig(
            use_legacy_sql=False,
        )

        query_job = client.query(
            executable_sql,
            job_config=query_job_config,
            location=settings.bigquery_location,
        )

        result_iterator = query_job.result()

        rows = [
            _serialize_row(dict(row))
            for row in result_iterator
        ]

        return {
            "success": True,
            "sql_query": executable_sql,
            "rows": rows,
            "row_count": len(rows),
            "job_id": query_job.job_id,
            "total_bytes_processed": query_job.total_bytes_processed,
            "error": None,
        }

    except GoogleAPIError as error:
        return {
            "success": False,
            "sql_query": executable_sql,
            "rows": [],
            "row_count": 0,
            "job_id": None,
            "total_bytes_processed": None,
            "error": f"{type(error).__name__}: {error}",
        }

    except Exception as error:
        return {
            "success": False,
            "sql_query": executable_sql,
            "rows": [],
            "row_count": 0,
            "job_id": None,
            "total_bytes_processed": None,
            "error": f"{type(error).__name__}: {error}",
        }


def _serialize_row(row: dict[str, Any]) -> dict[str, Any]:
    """Convert a BigQuery row dictionary into JSON-friendly values."""
    return {
        key: _serialize_value(value)
        for key, value in row.items()
    }


def _serialize_value(value: Any) -> Any:
    """Convert BigQuery/Python values into JSON-friendly values."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, Decimal):
        return float(value)

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    if isinstance(value, list):
        return [_serialize_value(item) for item in value]

    if isinstance(value, tuple):
        return [_serialize_value(item) for item in value]

    if isinstance(value, dict):
        return {
            str(key): _serialize_value(item)
            for key, item in value.items()
        }

    return value