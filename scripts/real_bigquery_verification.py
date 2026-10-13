"""Opt-in, bounded real-BigQuery integration verification.

The module is importable for fully offline testing.  Its CLI refuses to run
unless explicitly enabled and never persists preview rows or query results.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from google.cloud import bigquery
from langchain_core.messages import HumanMessage
from sqlglot import exp, parse_one

from agent.graph import build_graph
from app.config import settings
import db.bigquery_client as backend_bigquery
import agent.nodes.message_classification as classification_node
import agent.nodes.request_evidence as evidence_node
import agent.nodes.schema_mapping as mapping_node
from scripts.bigquery_preview_support import (
    REQUIRED_PREVIEW_FIELDS,
    decode_preview_row_by_name,
    select_preview_fields_in_physical_order,
)
from tools.schema_context_tool import get_schema_context_tool


RUNNER_VERSION = "1.0.0"
MAXIMUM_BYTES_BILLED = 100 * 1024 * 1024
PREVIEW_LIMIT = 100
QUERY_TIMEOUT_SECONDS = 30
EXPECTED_SCHEMA_KEYS = frozenset(
    {"schema_context", "semantic_context", "requirements_context", "public_schema_context"}
)
PLACEHOLDERS = {
    "your-gcp-project-id",
    "your_bigquery_dataset",
    "your_bigquery_table",
    "YOUR_BIGQUERY_LOCATION",
}
NUMERIC_TYPES = frozenset({"INTEGER", "INT64", "FLOAT", "FLOAT64", "NUMERIC", "BIGNUMERIC"})
TIME_TYPES = frozenset({"TIMESTAMP", "DATETIME"})


class VerificationError(RuntimeError):
    """A fail-closed verification error safe to summarize without its value."""


@dataclass(frozen=True)
class FrozenSelection:
    district: str
    utc_day: date

    @property
    def token(self) -> str:
        material = f"{self.district}\0{self.utc_day.isoformat()}".encode()
        return hashlib.sha256(material).hexdigest()[:12]


@dataclass
class QueryLedger:
    backend_limit: int = 2
    reference_limit: int = 2
    backend_live: int = 0
    reference_live: int = 0
    backend_dry_runs: int = 0
    reference_dry_runs: int = 0
    backend_processed_bytes: list[int] = field(default_factory=list)
    reference_processed_bytes: list[int] = field(default_factory=list)

    def reserve(self, category: str, *, dry_run: bool) -> None:
        if category not in {"backend", "reference"}:
            raise VerificationError("unknown query budget category")
        if dry_run:
            setattr(self, f"{category}_dry_runs", getattr(self, f"{category}_dry_runs") + 1)
            return
        live_name = f"{category}_live"
        limit = getattr(self, f"{category}_limit")
        if getattr(self, live_name) >= limit:
            raise VerificationError(f"{category} live SELECT budget exhausted")
        setattr(self, live_name, getattr(self, live_name) + 1)


@dataclass
class VerificationDependencies:
    schema_loader: Callable[[], dict[str, Any]] = get_schema_context_tool
    graph_factory: Callable[[], Any] = lambda: build_graph(use_checkpointer=False)
    client_factory: Callable[..., Any] = bigquery.Client


def validate_configuration(config: Any) -> None:
    values = (
        config.gcp_project_id,
        config.bigquery_dataset_id,
        config.bigquery_table_id,
        config.bigquery_location,
    )
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise VerificationError("required BigQuery configuration is absent")
    if any(value in PLACEHOLDERS for value in values):
        raise VerificationError("BigQuery configuration still uses public example placeholders")


def validate_context_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    missing = EXPECTED_SCHEMA_KEYS.difference(payload)
    if missing:
        raise VerificationError("schema loader contract missing key(s): " + ", ".join(sorted(missing)))
    for key in EXPECTED_SCHEMA_KEYS:
        if not isinstance(payload[key], dict):
            raise VerificationError(f"schema loader key {key} must contain a mapping")

    public = payload["public_schema_context"]
    metrics = public.get("queryable_metrics")
    if not isinstance(metrics, list) or not any(
        isinstance(item, dict)
        and item.get("concept") == "flow"
        and item.get("internal_column") == "value"
        for item in metrics
    ):
        raise VerificationError("public schema does not map flow to the value field")
    districts = (
        public.get("user_facing_concepts", {}).get("district", {}).get("values")
    )
    if not isinstance(districts, list) or not districts or not all(isinstance(item, str) for item in districts):
        raise VerificationError("public schema has no supported district catalog")

    requirements = payload["requirements_context"]
    defaults = requirements.get("defaults", {})
    if (
        defaults.get("table") != "sensor_measurements"
        or defaults.get("metric_column") != "value"
        or defaults.get("time_column") != "timestamp"
    ):
        raise VerificationError("requirements defaults do not match the supported table mapping")
    optional = requirements.get("requirements", {}).get("retrieve_data", {}).get("optional_slots")
    if not isinstance(optional, list) or "type" not in optional:
        raise VerificationError("measurement-type semantics are not explicit in requirements")

    logical_table = payload["schema_context"].get("tables", {}).get("sensor_measurements")
    if not isinstance(logical_table, dict):
        raise VerificationError("logical sensor_measurements catalog is absent")
    columns = logical_table.get("columns", {})
    expected_types = {"district": "string", "timestamp": "datetime", "value": "float"}
    if any(columns.get(name, {}).get("type") != kind for name, kind in expected_types.items()):
        raise VerificationError("logical preview field types do not match the supported contract")
    if "district" not in logical_table.get("allowed_filter_columns", []):
        raise VerificationError("district is not an allowed logical filter")
    return dict(payload)


def validate_table_schema(table: Any) -> list[Any]:
    fields = select_preview_fields_in_physical_order(table.schema, REQUIRED_PREVIEW_FIELDS)
    by_name = {field.name: str(field.field_type).upper() for field in fields}
    if by_name.get("district") != "STRING":
        raise VerificationError("configured district field is not STRING")
    if by_name.get("timestamp") not in TIME_TYPES:
        raise VerificationError("configured timestamp field has an unsupported type")
    if by_name.get("value") not in NUMERIC_TYPES:
        raise VerificationError("configured flow field has an unsupported numeric type")
    return fields


def _utc_datetime(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def select_nonempty_day(
    decoded_rows: Iterable[dict[str, Any]], supported_districts: Sequence[str]
) -> FrozenSelection:
    allowed = set(supported_districts)
    for row in decoded_rows:
        # Explicit None check preserves valid numeric zero measurements.
        if row["value"] is not None and row["district"] in allowed:
            return FrozenSelection(row["district"], _utc_datetime(row["timestamp"]).date())
    raise VerificationError("bounded preview did not contain a supported non-null flow selection")


def _select_only(sql: str) -> None:
    try:
        tree = parse_one(sql, read="bigquery")
    except Exception as exc:
        raise VerificationError("query could not be parsed as BigQuery SQL") from exc
    if not isinstance(tree, exp.Select):
        raise VerificationError("only a single SELECT is authorized")
    forbidden = (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop, exp.Alter, exp.Merge)
    if any(tree.find(kind) is not None for kind in forbidden):
        raise VerificationError("database modification statement rejected")


class _CompletedJob:
    def __init__(self, job: Any, rows: list[Any]):
        self._job = job
        self._rows = rows
        self.job_id = getattr(job, "job_id", None)
        self.total_bytes_processed = int(getattr(job, "total_bytes_processed", 0) or 0)

    def result(self, *args: Any, **kwargs: Any) -> list[Any]:
        return self._rows


class GuardedBackendClient:
    """Client facade used by the unchanged backend execution wrapper."""

    def __init__(self, client: Any, ledger: QueryLedger, location: str):
        self._client = client
        self._ledger = ledger
        self._location = location

    def query(self, sql: str, **_ignored: Any) -> _CompletedJob:
        _select_only(sql)
        dry_config = bigquery.QueryJobConfig(
            dry_run=True,
            use_query_cache=False,
            use_legacy_sql=False,
            maximum_bytes_billed=MAXIMUM_BYTES_BILLED,
        )
        self._ledger.reserve("backend", dry_run=True)
        dry = self._client.query(
            sql,
            job_config=dry_config,
            location=self._location,
            retry=None,
            job_retry=None,
            timeout=QUERY_TIMEOUT_SECONDS,
        )
        processed = int(getattr(dry, "total_bytes_processed", 0) or 0)
        if processed > MAXIMUM_BYTES_BILLED:
            raise VerificationError("backend dry run exceeds the 100 MiB cap")

        self._ledger.reserve("backend", dry_run=False)
        live_config = bigquery.QueryJobConfig(
            use_query_cache=False,
            use_legacy_sql=False,
            maximum_bytes_billed=MAXIMUM_BYTES_BILLED,
        )
        live = self._client.query(
            sql,
            job_config=live_config,
            location=self._location,
            retry=None,
            job_retry=None,
            timeout=QUERY_TIMEOUT_SECONDS,
        )
        rows = list(live.result(timeout=QUERY_TIMEOUT_SECONDS, retry=None))
        self._ledger.backend_processed_bytes.append(
            int(getattr(live, "total_bytes_processed", 0) or 0)
        )
        return _CompletedJob(live, rows)


@contextmanager
def _backend_client_override(client: Any, ledger: QueryLedger, location: str) -> Iterator[None]:
    original = backend_bigquery.bigquery.Client
    backend_bigquery.bigquery.Client = lambda *args, **kwargs: GuardedBackendClient(client, ledger, location)
    try:
        yield
    finally:
        backend_bigquery.bigquery.Client = original


def _reference_sql(
    selection: FrozenSelection, aggregate: bool, physical_table_name: str
) -> tuple[str, list[Any]]:
    end = selection.utc_day + timedelta(days=1)
    projection = "SAFE_DIVIDE(SUM(value), COUNT(value)) AS reference_average" if aggregate else "timestamp, value"
    suffix = "LIMIT 1" if aggregate else "ORDER BY timestamp ASC, value ASC LIMIT 100"
    sql = (
        f"SELECT {projection} FROM {physical_table_name} "
        "WHERE district = @district AND timestamp >= @start_time AND timestamp < @end_time "
        f"{suffix}"
    )
    params = [
        bigquery.ScalarQueryParameter("district", "STRING", selection.district),
        bigquery.ScalarQueryParameter("start_time", "TIMESTAMP", datetime.combine(selection.utc_day, datetime.min.time(), timezone.utc)),
        bigquery.ScalarQueryParameter("end_time", "TIMESTAMP", datetime.combine(end, datetime.min.time(), timezone.utc)),
    ]
    return sql, params


class _CountingChain:
    def __init__(self, chain: Any, counter: list[int]):
        self._chain = chain
        self._counter = counter

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        self._counter[0] += 1
        return self._chain.invoke(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chain, name)


@contextmanager
def _count_model_calls() -> Iterator[list[int]]:
    counter = [0]
    originals = (
        classification_node.get_message_classification_chain,
        evidence_node.get_request_evidence_chain,
        mapping_node.get_schema_mapping_chain,
    )

    def wrap(factory: Callable[..., Any]) -> Callable[..., Any]:
        return lambda *args, **kwargs: _CountingChain(factory(*args, **kwargs), counter)

    classification_node.get_message_classification_chain = wrap(originals[0])
    evidence_node.get_request_evidence_chain = wrap(originals[1])
    mapping_node.get_schema_mapping_chain = wrap(originals[2])
    try:
        yield counter
    finally:
        classification_node.get_message_classification_chain = originals[0]
        evidence_node.get_request_evidence_chain = originals[1]
        mapping_node.get_schema_mapping_chain = originals[2]


def _run_reference(client: Any, sql: str, params: list[Any], ledger: QueryLedger, location: str) -> list[dict[str, Any]]:
    _select_only(sql)
    dry_config = bigquery.QueryJobConfig(
        dry_run=True, use_query_cache=False, use_legacy_sql=False,
        maximum_bytes_billed=MAXIMUM_BYTES_BILLED, query_parameters=params,
    )
    ledger.reserve("reference", dry_run=True)
    dry = client.query(
        sql, job_config=dry_config, location=location, retry=None, job_retry=None,
        timeout=QUERY_TIMEOUT_SECONDS,
    )
    if int(getattr(dry, "total_bytes_processed", 0) or 0) > MAXIMUM_BYTES_BILLED:
        raise VerificationError("reference dry run exceeds the 100 MiB cap")
    ledger.reserve("reference", dry_run=False)
    live_config = bigquery.QueryJobConfig(
        use_query_cache=False, use_legacy_sql=False,
        maximum_bytes_billed=MAXIMUM_BYTES_BILLED, query_parameters=params,
    )
    live = client.query(
        sql, job_config=live_config, location=location, retry=None, job_retry=None,
        timeout=QUERY_TIMEOUT_SECONDS,
    )
    rows = [dict(row) for row in live.result(timeout=QUERY_TIMEOUT_SECONDS, retry=None)]
    ledger.reference_processed_bytes.append(int(getattr(live, "total_bytes_processed", 0) or 0))
    return rows


def _raw_pairs(rows: Iterable[Mapping[str, Any]]) -> list[tuple[datetime, float]]:
    pairs = [(datetime.fromisoformat(str(row["timestamp"])), float(row["value"])) for row in rows]
    return sorted(pairs, key=lambda item: (item[0], item[1]))


def compare_results(raw_state: Mapping[str, Any], avg_state: Mapping[str, Any], raw_reference: list[dict[str, Any]], avg_reference: list[dict[str, Any]]) -> None:
    raw_result = raw_state.get("query_result") or {}
    avg_result = avg_state.get("query_result") or {}
    if not raw_result.get("success") or int(raw_result.get("row_count", 0)) <= 0:
        raise VerificationError("backend raw retrieval was empty or unsuccessful")
    if _raw_pairs(raw_result.get("rows") or []) != _raw_pairs(raw_reference):
        raise VerificationError("backend raw retrieval did not match the independent reference")
    backend_rows = avg_result.get("rows") or []
    if not avg_result.get("success") or len(backend_rows) != 1 or len(avg_reference) != 1:
        raise VerificationError("average result cardinality was not one")
    left = backend_rows[0].get("avg_value")
    right = avg_reference[0].get("reference_average")
    if left is None or right is None or not math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-9):
        raise VerificationError("average was null or did not match the independent reference")
    for state in (raw_state, avg_state):
        answer = state.get("final_answer")
        if not isinstance(answer, str) or "query executed successfully" not in answer.lower() or "Executed SQL:" not in answer:
            raise VerificationError("backend response formatting was incomplete")


def _require_graph_execution(state: Mapping[str, Any], category: str) -> None:
    result = state.get("query_result") or {}
    if state.get("execution_error") or not result.get("success"):
        raise VerificationError(f"{category} graph execution failed")
    if not state.get("sql_is_safe"):
        raise VerificationError(f"{category} graph SQL did not pass mandatory safety validation")


def run_verification(
    *,
    config: Any = settings,
    dependencies: VerificationDependencies | None = None,
    tested_commit: str,
    attempt_label: str,
) -> dict[str, Any]:
    deps = dependencies or VerificationDependencies()
    validate_configuration(config)
    contexts = validate_context_contract(deps.schema_loader())

    client = deps.client_factory(project=config.gcp_project_id, location=config.bigquery_location)
    table = client.get_table(f"{config.gcp_project_id}.{config.bigquery_dataset_id}.{config.bigquery_table_id}", retry=None, timeout=QUERY_TIMEOUT_SECONDS)
    selected_fields = validate_table_schema(table)
    table_location = getattr(table, "location", None)
    if table_location and str(table_location).upper() != str(config.bigquery_location).upper():
        raise VerificationError("configured location does not match table metadata")

    preview = client.list_rows(
        table,
        selected_fields=selected_fields,
        max_results=PREVIEW_LIMIT,
        retry=None,
        timeout=QUERY_TIMEOUT_SECONDS,
    )
    decoded = [decode_preview_row_by_name(row) for row in preview]
    districts = contexts["public_schema_context"]["user_facing_concepts"]["district"]["values"]
    selection = select_nonempty_day(decoded, districts)

    raw_request = f"Show up to 100 water flow measurements in {selection.district} on {selection.utc_day.isoformat()}, ordered by time."
    avg_request = f"Show the average water flow in {selection.district} on {selection.utc_day.isoformat()}."
    physical_table_name = (
        f"`{config.gcp_project_id}.{config.bigquery_dataset_id}.{config.bigquery_table_id}`"
    )
    raw_reference_sql, raw_params = _reference_sql(
        selection, aggregate=False, physical_table_name=physical_table_name
    )
    avg_reference_sql, avg_params = _reference_sql(
        selection, aggregate=True, physical_table_name=physical_table_name
    )

    ledger = QueryLedger()
    with _count_model_calls() as model_counter:
        graph = deps.graph_factory()
        with _backend_client_override(client, ledger, config.bigquery_location):
            raw_state = graph.invoke({"messages": [HumanMessage(content=raw_request)]})
            _require_graph_execution(raw_state, "raw")
            avg_state = graph.invoke({"messages": [HumanMessage(content=avg_request)]})
            _require_graph_execution(avg_state, "average")
    raw_reference = _run_reference(client, raw_reference_sql, raw_params, ledger, config.bigquery_location)
    avg_reference = _run_reference(client, avg_reference_sql, avg_params, ledger, config.bigquery_location)
    compare_results(raw_state, avg_state, raw_reference, avg_reference)

    return {
        "attempt_label": attempt_label,
        "runner_version": RUNNER_VERSION,
        "tested_commit": tested_commit,
        "selection_token": selection.token,
        "preview_calls": 1,
        "preview_rows_scanned": len(decoded),
        "model_calls": model_counter[0],
        "backend_dry_runs": ledger.backend_dry_runs,
        "backend_live_selects": ledger.backend_live,
        "reference_dry_runs": ledger.reference_dry_runs,
        "reference_live_selects": ledger.reference_live,
        "backend_processed_bytes": ledger.backend_processed_bytes,
        "reference_processed_bytes": ledger.reference_processed_bytes,
        "raw_comparison": "PASS",
        "average_comparison": "PASS",
        "zero_database_modifications": True,
    }


def installed_ollama_tags() -> list[str]:
    completed = subprocess.run(
        ["ollama", "list"], check=True, capture_output=True, text=True, timeout=10
    )
    tags: list[str] = []
    for line in completed.stdout.splitlines()[1:]:
        parts = line.split()
        if parts:
            tags.append(parts[0])
    return tags


def validate_live_model_roles(config: Any) -> None:
    roles = ("interpret", "mapping", "clarification", "sql", "answer")
    for role in roles:
        if getattr(config, f"{role}_model") != "llama3.2:latest":
            raise VerificationError("active model roles are not pinned to the authorized installed tag")
        if float(getattr(config, f"{role}_temperature")) != 0.0:
            raise VerificationError("active model role temperatures are not all zero")


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-label", required=True)
    parser.add_argument("--tested-commit", required=True)
    parser.add_argument("--private-report", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("RUN_REAL_BIGQUERY_VERIFICATION") != "1":
        raise SystemExit("Refusing live verification: set RUN_REAL_BIGQUERY_VERIFICATION=1 explicitly")
    os.environ["AGENT_DEBUG"] = "false"
    os.environ["LANGSMITH_TRACING"] = "false"
    if "llama3.2:latest" not in installed_ollama_tags():
        raise SystemExit("Required installed model tag llama3.2:latest is unavailable")
    validate_live_model_roles(settings)
    args.private_report.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = run_verification(tested_commit=args.tested_commit, attempt_label=args.attempt_label)
    except Exception as exc:
        private_failure = {
            "attempt_label": args.attempt_label,
            "runner_version": RUNNER_VERSION,
            "tested_commit": args.tested_commit,
            "outcome": "FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "zero_database_modifications": True,
        }
        args.private_report.write_text(json.dumps(private_failure, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({
            "attempt_label": args.attempt_label,
            "runner_version": RUNNER_VERSION,
            "outcome": "FAIL",
            "error_type": type(exc).__name__,
            "zero_database_modifications": True,
        }, indent=2))
        return 1
    args.private_report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "selection_token"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
