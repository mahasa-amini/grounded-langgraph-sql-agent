import re
from typing import Any


_AGGREGATION_SQL_FUNCTIONS = {
    "avg": "AVG",
    "average": "AVG",
    "mean": "AVG",
    "sum": "SUM",
    "total": "SUM",
    "min": "MIN",
    "minimum": "MIN",
    "max": "MAX",
    "maximum": "MAX",
    "count": "COUNT",
}

_ALLOWED_HAVING_OPERATORS = {
    "=",
    "!=",
    ">",
    ">=",
    "<",
    "<=",
}

_SAFE_COLUMN_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def enforce_distinct_select_tool(
    sql_query: str,
    distinct: bool,
    aggregation: str | None,
) -> str:
    """
    Enforce SELECT DISTINCT for non-aggregation queries when the mapped state
    explicitly requests distinct output.
    """

    if not sql_query:
        return sql_query

    if not distinct:
        return sql_query

    if aggregation:
        return sql_query

    if re.search(r"^\s*SELECT\s+DISTINCT\b", sql_query, flags=re.IGNORECASE):
        return sql_query

    return re.sub(
        pattern=r"^\s*SELECT\s+",
        repl="SELECT DISTINCT ",
        string=sql_query,
        count=1,
        flags=re.IGNORECASE,
    )


def enforce_having_conditions_tool(
    sql_query: str,
    having_conditions: list[dict[str, Any]] | None,
    aggregation: str | None,
    metric_column: str | None,
) -> str:
    """
    Deterministically enforce HAVING conditions for aggregation queries.

    This prevents the LLM from:
    - forgetting HAVING
    - putting aggregate thresholds in WHERE
    - using unstable aliases such as avg_value in HAVING
    """

    if not sql_query:
        return sql_query

    if not aggregation:
        return sql_query

    if not metric_column:
        return sql_query

    if not having_conditions:
        return sql_query

    having_clause = _build_having_clause(
        having_conditions=having_conditions,
        aggregation=aggregation,
        metric_column=metric_column,
    )

    if not having_clause:
        return sql_query

    return _replace_or_insert_having_clause(
        sql_query=sql_query,
        having_clause=having_clause,
    )


def _build_having_clause(
    having_conditions: list[dict[str, Any]],
    aggregation: str,
    metric_column: str,
) -> str | None:
    aggregation_function = _resolve_aggregation_function(aggregation)

    if aggregation_function is None:
        return None

    if not _is_safe_column_name(metric_column):
        return None

    expressions: list[str] = []

    for condition in having_conditions:
        if not isinstance(condition, dict):
            continue

        condition_column = condition.get("column")

        if condition_column and condition_column != metric_column:
            continue

        operator = str(condition.get("operator") or "").strip()

        if operator not in _ALLOWED_HAVING_OPERATORS:
            continue

        value_expression = _format_numeric_value(condition.get("value"))

        if value_expression is None:
            continue

        expressions.append(
            f"{aggregation_function}({metric_column}) {operator} {value_expression}"
        )

    if not expressions:
        return None

    return "HAVING " + " AND ".join(expressions)


def _resolve_aggregation_function(aggregation: str) -> str | None:
    normalized_aggregation = str(aggregation or "").strip().lower()

    return _AGGREGATION_SQL_FUNCTIONS.get(normalized_aggregation)


def _format_numeric_value(value: Any) -> str | None:
    if isinstance(value, bool):
        return None

    if isinstance(value, int | float):
        return str(value)

    if isinstance(value, str):
        cleaned_value = value.strip()

        if re.fullmatch(r"-?\d+(?:\.\d+)?", cleaned_value):
            return cleaned_value

    return None


def _replace_or_insert_having_clause(
    sql_query: str,
    having_clause: str,
) -> str:
    stripped_sql = sql_query.strip().rstrip(";")

    existing_having_match = re.search(
        r"\bHAVING\b",
        stripped_sql,
        flags=re.IGNORECASE,
    )

    if existing_having_match:
        prefix = stripped_sql[: existing_having_match.start()].rstrip()
        after_having = stripped_sql[existing_having_match.end() :]

        next_clause_match = re.search(
            r"\b(ORDER\s+BY|LIMIT)\b",
            after_having,
            flags=re.IGNORECASE,
        )

        if next_clause_match:
            suffix = after_having[next_clause_match.start() :].strip()
            return f"{prefix} {having_clause} {suffix}".strip()

        return f"{prefix} {having_clause}".strip()

    insertion_match = re.search(
        r"\b(ORDER\s+BY|LIMIT)\b",
        stripped_sql,
        flags=re.IGNORECASE,
    )

    if insertion_match:
        prefix = stripped_sql[: insertion_match.start()].rstrip()
        suffix = stripped_sql[insertion_match.start() :].strip()

        return f"{prefix} {having_clause} {suffix}".strip()

    return f"{stripped_sql} {having_clause}".strip()


def _is_safe_column_name(column_name: str) -> bool:
    return bool(_SAFE_COLUMN_PATTERN.fullmatch(column_name))