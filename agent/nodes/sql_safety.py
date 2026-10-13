from typing import Any
import re
from langchain_core.messages import AIMessage
from agent.state import SQLAgentState
from agent.utils.debug import debug_node
from tools.sql_safety_guard_tool import validate_sql_safety_tool


READ_ONLY_SQL_SAFETY_ERROR = (
    "This operation is not allowed. I can only help generate read-only SELECT queries."
)


UNSAFE_SQL_PATTERNS: list[tuple[str, str]] = [
    ("DROP statement", r"\bdrop\s+(table|schema|database|view|materialized\s+view)\b"),
    ("TRUNCATE statement", r"\btruncate\s+(table\s+)?[a-zA-Z_][a-zA-Z0-9_.$`]*\b"),
    ("DELETE statement", r"\bdelete\s+from\b"),
    ("UPDATE statement", r"\bupdate\s+[a-zA-Z_][a-zA-Z0-9_.$`]*\s+set\b"),
    ("INSERT statement", r"\binsert\s+into\b"),
    ("ALTER statement", r"\balter\s+(table|schema|database|view|materialized\s+view)\b"),
    ("CREATE statement", r"\bcreate\s+(or\s+replace\s+)?(table|schema|database|view|materialized\s+view)\b"),
    ("MERGE statement", r"\bmerge\s+into\b"),
    ("REPLACE statement", r"\breplace\s+(table|view)\b"),
    ("GRANT statement", r"\bgrant\b.+\bon\b"),
    ("REVOKE statement", r"\brevoke\b.+\bon\b"),
    ("EXECUTE IMMEDIATE statement", r"\bexecute\s+immediate\b"),
    ("CALL statement", r"\bcall\s+[a-zA-Z_][a-zA-Z0-9_.$`]*\b"),
    (
        "destructive natural-language request",
        r"\b(delete|remove|erase|clear|wipe|destroy)\s+(all\s+)?(data|records|rows|table|tables|database|dataset)\b",
    ),
]


def normalize_text_for_early_sql_safety(value: Any) -> str:
    text = str(value or "")
    return " ".join(text.strip().split())


def detect_unsafe_sql_operation(user_text: str) -> dict[str, Any] | None:
    normalized_text = normalize_text_for_early_sql_safety(user_text)

    if not normalized_text:
        return None

    for reason, pattern in UNSAFE_SQL_PATTERNS:
        match = re.search(
            pattern=pattern,
            string=normalized_text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if match is None:
            continue

        return {
            "reason": reason,
            "matched_text": match.group(0),
            "pattern": pattern,
        }

    return None


def is_early_sql_safety_error(error: Any) -> bool:
    return str(error or "").strip() == READ_ONLY_SQL_SAFETY_ERROR


def early_sql_safety_check_node(state: SQLAgentState) -> dict[str, Any]:
    latest_user_message = state.get("latest_user_message", "")
    user_text = normalize_text_for_early_sql_safety(latest_user_message)

    violation = detect_unsafe_sql_operation(user_text)

    debug_node(
        "EARLY SQL SAFETY CHECK",
        {
            "user_text": user_text,
            "violation": violation,
        },
    )

    if violation is None:
        return {
            "error": None,
        }

    return {
        "request_status": "idle",
        "active_request_type": None,
        "active_request_text": None,
        "mapping_input_text": "",
        "clarification_answers": [],
        "missing_slots": [],
        "request_evidence": {},
        "evidence_suggestions": {},
        "table_name": None,
        "raw_metric": None,
        "metric_column": None,
        "time_column": None,
        "time_range": None,
        "filters": [],
        "selected_columns": [],
        "aggregation": None,
        "group_by": [],
        "order_by": None,
        "limit": None,
        "unsupported_terms": [],
        "ambiguous_terms": [],
        "error": READ_ONLY_SQL_SAFETY_ERROR,
    }


def security_reject_message_node(state: SQLAgentState) -> dict[str, Any]:
    """
    Return a user-friendly message for unsafe SQL/data operations.
    """

    error = state.get("error")
    sql_safety_error = state.get("sql_safety_error")
    reason = error or sql_safety_error

    if is_early_sql_safety_error(reason):
        final_answer = (
            "I can’t perform destructive database operations. "
            "I can only help with read-only SELECT queries."
        )
    elif reason:
        final_answer = (
            "I can’t complete this request because it is not a safe read-only data query. "
            "I can only help with read-only SELECT queries."
        )
    else:
        final_answer = (
            "I can’t complete this request because it may modify or damage the database. "
            "I can only help with read-only SELECT queries."
        )

    return {
        "messages": [AIMessage(content=final_answer)],
        "final_answer": final_answer,
        "request_status": "idle",
        "active_request_type": None,
        "active_request_text": None,
        "mapping_input_text": "",
        "clarification_answers": [],
        "missing_slots": [],
        "error": None,
    }

def validate_sql_safety_node(state: SQLAgentState) -> SQLAgentState:
    requirements_context: dict[str, Any] = state.get("requirements_context") or {}
    defaults = requirements_context.get("defaults", {})
    configured_table = defaults.get("table") if isinstance(defaults, dict) else None

    schema_context = state.get("schema_context") or {}
    schema_tables = schema_context.get("tables", {}) if isinstance(schema_context, dict) else {}
    permitted_table = (
        configured_table
        if isinstance(configured_table, str)
        and isinstance(schema_tables, dict)
        and configured_table in schema_tables
        else None
    )

    validation_result = validate_sql_safety_tool(
        sql_query=state.get("sql_query"),
        requirements_context=requirements_context,
        permitted_table=permitted_table,
    )

    return {
        "sql_is_safe": validation_result["is_safe"],
        "sql_validation_errors": validation_result["errors"],
    }
