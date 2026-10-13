from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any

from sqlglot import exp, parse_one

from tools.normalize_time_range_tool import build_bigquery_time_predicate_tool


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ORDER_BY = re.compile(
    r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?:\s+(?P<direction>ASC|DESC))?$",
    flags=re.IGNORECASE,
)
_AGGREGATION_FUNCTIONS = {
    "avg": "AVG",
    "sum": "SUM",
    "min": "MIN",
    "max": "MAX",
    "count": "COUNT",
}
_COMPARISON_EXPRESSIONS = {
    "=": exp.EQ,
    "!=": exp.NEQ,
    ">": exp.GT,
    ">=": exp.GTE,
    "<": exp.LT,
    "<=": exp.LTE,
}


def compile_validated_sql_plan_tool(state: dict[str, Any]) -> dict[str, Any]:
    """Compile the project's validated query state into BigQuery SQL.

    Only typed fields already represented by SQLAgentState are accepted. The
    function deliberately rejects expressions embedded in identifiers, order
    strings, filter values, and grouping fields.
    """

    errors: list[str] = []
    schema_context = _require_dict(state.get("schema_context"), "schema_context", errors)
    requirements = _require_dict(
        state.get("requirements_context"), "requirements_context", errors
    )
    tables = schema_context.get("tables") if isinstance(schema_context, dict) else None
    if not isinstance(tables, dict):
        errors.append("schema_context.tables must be a mapping.")
        tables = {}

    table_name = state.get("table_name")
    if not isinstance(table_name, str) or not _IDENTIFIER.fullmatch(table_name):
        errors.append("table_name must be a validated logical identifier.")
        table_name = None
    elif table_name not in tables:
        errors.append(f"Unknown logical table: {table_name}.")

    table_data = tables.get(table_name, {}) if table_name else {}
    columns = table_data.get("columns") if isinstance(table_data, dict) else None
    if not isinstance(columns, dict) or not columns:
        errors.append("The logical table must declare typed columns.")
        columns = {}
    available_columns = set(columns)

    metric_column = _validated_column(
        state.get("metric_column"), available_columns, "metric_column", errors
    )
    time_column = _validated_column(
        state.get("time_column"), available_columns, "time_column", errors
    )
    if time_column and _column_type(columns, time_column) != "datetime":
        errors.append("time_column must reference a datetime column.")

    selected_columns = _validated_column_list(
        state.get("selected_columns"), available_columns, "selected_columns", errors
    )
    group_by = _validated_column_list(
        state.get("group_by"), available_columns, "group_by", errors
    )
    allowed_group_by = set(table_data.get("allowed_group_by_columns", []))
    for column in group_by:
        if column not in allowed_group_by:
            errors.append(f"Grouping is not allowed for column: {column}.")
    if state.get("time_bucket") is not None:
        errors.append(
            "time_bucket is not a supported typed field in the current request contract."
        )

    aggregation = state.get("aggregation")
    supported_aggregations = set(requirements.get("aggregations", {}).get("supported", []))
    if aggregation is not None:
        if not isinstance(aggregation, str) or aggregation not in supported_aggregations:
            errors.append(f"Unsupported aggregation: {aggregation!r}.")
        elif aggregation not in _AGGREGATION_FUNCTIONS:
            errors.append(f"Aggregation has no compiler implementation: {aggregation}.")
        elif metric_column and _column_type(columns, metric_column) not in {"float", "integer"}:
            errors.append("Aggregation requires a numeric metric column.")

    distinct = state.get("distinct", False)
    if type(distinct) is not bool:
        errors.append("distinct must be a boolean.")
        distinct = False

    if aggregation:
        if distinct:
            errors.append("distinct cannot be combined with aggregation.")
        if selected_columns != group_by:
            errors.append(
                "For aggregation, selected_columns must exactly match group_by; "
                "the metric is emitted only through the aggregate expression."
            )
    else:
        if group_by:
            errors.append("group_by requires an aggregation.")
        if not selected_columns:
            errors.append("A raw query requires at least one selected column.")

    filters = _compile_filters(
        raw_filters=state.get("filters"),
        columns=columns,
        allowed_columns=set(table_data.get("allowed_filter_columns", [])),
        supported_operators=set(requirements.get("filters", {}).get("supported_operators", [])),
        time_column=time_column,
        has_time_range=state.get("time_range") is not None,
        errors=errors,
    )
    _validate_group_filter_conflicts(state.get("filters"), group_by, errors)
    time_expression = _compile_time_range(
        state.get("time_range"), time_column, errors
    )

    having_conditions = _compile_having(
        raw_conditions=state.get("having_conditions"),
        aggregation=aggregation if isinstance(aggregation, str) else None,
        metric_column=metric_column,
        errors=errors,
    )

    order_expression = _compile_order_by(
        state.get("order_by"),
        aggregation=aggregation if isinstance(aggregation, str) else None,
        group_by=group_by,
        selected_columns=selected_columns,
        allowed_raw_columns=set(table_data.get("allowed_order_by_columns", [])),
        errors=errors,
    )
    limit = _validate_limit(state.get("limit"), requirements, errors)

    unsupported_terms = state.get("unsupported_terms", [])
    ambiguous_terms = state.get("ambiguous_terms", [])
    if not isinstance(unsupported_terms, list) or unsupported_terms:
        errors.append("A compilable plan cannot contain unsupported terms.")
    if not isinstance(ambiguous_terms, list) or ambiguous_terms:
        errors.append("A compilable plan cannot contain ambiguous terms.")

    if errors:
        return {"sql_query": None, "errors": list(dict.fromkeys(errors)), "plan": None}

    assert table_name is not None
    assert metric_column is not None
    select_expressions: list[exp.Expression]
    aggregate_alias = None
    if aggregation:
        aggregate_alias = f"{aggregation}_value"
        aggregate = exp.func(
            _AGGREGATION_FUNCTIONS[aggregation], exp.column(metric_column)
        ).as_(aggregate_alias)
        select_expressions = [*[exp.column(column) for column in group_by], aggregate]
    else:
        select_expressions = [exp.column(column) for column in selected_columns]

    query = exp.select(*select_expressions).from_(exp.to_table(table_name))
    if distinct:
        query = query.distinct()

    predicates = [*filters]
    if time_expression is not None:
        predicates.append(time_expression)
    if predicates:
        predicate = predicates[0]
        for item in predicates[1:]:
            predicate = exp.and_(predicate, item)
        query = query.where(predicate)

    if group_by:
        query = query.group_by(*[exp.column(column) for column in group_by])
    if having_conditions:
        predicate = having_conditions[0]
        for item in having_conditions[1:]:
            predicate = exp.and_(predicate, item)
        query = query.having(predicate)
    if order_expression is not None:
        query = query.order_by(order_expression)
    if limit is not None:
        query = query.limit(limit)

    sql_query = query.sql(dialect="bigquery")
    return {
        "sql_query": sql_query,
        "errors": [],
        "plan": {
            "table_name": table_name,
            "selected_columns": selected_columns,
            "metric_column": metric_column,
            "time_column": time_column,
            "aggregation": aggregation,
            "aggregate_alias": aggregate_alias,
            "group_by": group_by,
            "order_by": state.get("order_by"),
            "distinct": distinct,
            "limit": limit,
            "filter_count": len(filters),
            "has_time_predicate": time_expression is not None,
            "having_count": len(having_conditions),
        },
    }


def _require_dict(value: Any, name: str, errors: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        errors.append(f"{name} must be a mapping.")
        return {}
    return value


def _validated_column(
    value: Any, available: set[str], name: str, errors: list[str]
) -> str | None:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        errors.append(f"{name} must be a validated column identifier.")
        return None
    if value not in available:
        errors.append(f"Unknown {name}: {value}.")
        return None
    return value


def _validated_column_list(
    value: Any, available: set[str], name: str, errors: list[str]
) -> list[str]:
    if not isinstance(value, list):
        errors.append(f"{name} must be a list of column identifiers.")
        return []
    result: list[str] = []
    for item in value:
        column = _validated_column(item, available, name, errors)
        if column and column not in result:
            result.append(column)
        elif column:
            errors.append(f"Duplicate {name} column: {column}.")
    return result


def _column_type(columns: dict[str, Any], column: str) -> str | None:
    data = columns.get(column)
    return data.get("type") if isinstance(data, dict) else None


def _compile_filters(
    raw_filters: Any,
    columns: dict[str, Any],
    allowed_columns: set[str],
    supported_operators: set[str],
    time_column: str | None,
    has_time_range: bool,
    errors: list[str],
) -> list[exp.Expression]:
    if not isinstance(raw_filters, list):
        errors.append("filters must be a list.")
        return []
    expressions: list[exp.Expression] = []
    for index, item in enumerate(raw_filters):
        if not isinstance(item, dict):
            errors.append(f"Filter {index} must be a typed mapping.")
            continue
        column = item.get("column")
        operator = item.get("operator")
        if not isinstance(column, str) or not _IDENTIFIER.fullmatch(column):
            errors.append(f"Filter {index} has an invalid column identifier.")
            continue
        if column not in columns or column not in allowed_columns:
            errors.append(f"Filter column is not allowed: {column}.")
            continue
        if has_time_range and column == time_column:
            errors.append("A time_range cannot be combined with a separate time-column filter.")
            continue
        if not isinstance(operator, str) or operator not in supported_operators:
            errors.append(f"Unsupported filter operator: {operator!r}.")
            continue
        column_expression = exp.column(column)
        value = item.get("value")
        column_type = _column_type(columns, column)
        if operator == "IN":
            if not isinstance(value, list) or not value:
                errors.append(f"IN filter {index} requires a non-empty list.")
                continue
            literals = [
                _literal(item_value, column_type, f"Filter {index}", errors)
                for item_value in value
            ]
            if any(literal is None for literal in literals):
                continue
            expressions.append(
                exp.In(this=column_expression, expressions=literals)
            )
            continue
        if operator == "BETWEEN":
            if not isinstance(value, list) or len(value) != 2:
                errors.append(f"BETWEEN filter {index} requires exactly two values.")
                continue
            low = _literal(value[0], column_type, f"Filter {index}", errors)
            high = _literal(value[1], column_type, f"Filter {index}", errors)
            if low is not None and high is not None:
                expressions.append(exp.Between(this=column_expression, low=low, high=high))
            continue
        if operator not in _COMPARISON_EXPRESSIONS:
            errors.append(f"Filter operator has no compiler implementation: {operator}.")
            continue
        literal = _literal(value, column_type, f"Filter {index}", errors)
        if literal is not None:
            expressions.append(
                _COMPARISON_EXPRESSIONS[operator](
                    this=column_expression, expression=literal
                )
            )
    return expressions


def _validate_group_filter_conflicts(
    raw_filters: Any, group_by: list[str], errors: list[str]
) -> None:
    if not isinstance(raw_filters, list):
        return
    fixed_columns = {
        item.get("column")
        for item in raw_filters
        if isinstance(item, dict) and item.get("operator") == "="
    }
    for column in group_by:
        if column in fixed_columns:
            errors.append(
                f"group_by column {column} conflicts with a single-value equality filter."
            )


def _literal(
    value: Any, column_type: str | None, label: str, errors: list[str]
) -> exp.Expression | None:
    if column_type == "string":
        if not isinstance(value, str):
            errors.append(f"{label} requires a string value.")
            return None
        return exp.Literal.string(value)
    if column_type in {"float", "integer"}:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"{label} requires a numeric value.")
            return None
        if not math.isfinite(float(value)):
            errors.append(f"{label} requires a finite numeric value.")
            return None
        return exp.Literal.number(str(value))
    if column_type == "datetime":
        if not isinstance(value, str) or _parse_datetime(value) is None:
            errors.append(f"{label} requires an ISO datetime value.")
            return None
        return exp.func("TIMESTAMP", exp.Literal.string(value))
    errors.append(f"{label} targets an unsupported column type: {column_type!r}.")
    return None


def _compile_time_range(
    value: Any, time_column: str | None, errors: list[str]
) -> exp.Expression | None:
    if not isinstance(value, dict):
        errors.append("time_range must be a normalized mapping.")
        return None
    kind = value.get("kind")
    if kind == "absolute":
        start = value.get("start")
        end = value.get("end")
        parsed_start = _parse_datetime(start) if isinstance(start, str) else None
        parsed_end = _parse_datetime(end) if isinstance(end, str) else None
        if parsed_start is None and parsed_end is None:
            errors.append("An absolute time_range requires a valid start or end.")
            return None
        if start is not None and parsed_start is None:
            errors.append("Absolute time_range start is invalid.")
            return None
        if end is not None and parsed_end is None:
            errors.append("Absolute time_range end is invalid.")
            return None
        if parsed_start and parsed_end:
            try:
                invalid_order = parsed_start >= parsed_end
            except TypeError:
                errors.append("Absolute time_range boundaries must use compatible timezones.")
                return None
            if invalid_order:
                errors.append("Absolute time_range start must be before its exclusive end.")
                return None
    elif kind == "relative":
        amount = value.get("amount")
        unit = value.get("unit")
        if type(amount) is not int or amount < 1:
            errors.append("Relative time_range amount must be a positive integer.")
            return None
        if unit not in {"SECOND", "MINUTE", "HOUR", "DAY"}:
            errors.append("Relative time_range unit is unsupported.")
            return None
    elif kind == "named":
        if value.get("name") not in {"today", "yesterday"}:
            errors.append("Named time_range is unsupported.")
            return None
    else:
        errors.append("time_range kind is unsupported or missing.")
        return None
    predicate = build_bigquery_time_predicate_tool(value, time_column)
    if not predicate:
        errors.append("The normalized time_range could not be compiled.")
        return None
    try:
        return parse_one(predicate, read="bigquery")
    except Exception:
        errors.append("The normalized time predicate could not be parsed.")
        return None


def _parse_datetime(value: str) -> datetime | None:
    try:
        normalized = value.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized)
    except (TypeError, ValueError):
        return None


def _compile_having(
    raw_conditions: Any,
    aggregation: str | None,
    metric_column: str | None,
    errors: list[str],
) -> list[exp.Expression]:
    if not isinstance(raw_conditions, list):
        errors.append("having_conditions must be a list.")
        return []
    if raw_conditions and not aggregation:
        errors.append("HAVING conditions require an aggregation.")
        return []
    if not aggregation or not metric_column:
        return []
    expressions: list[exp.Expression] = []
    aggregate_expression = exp.func(
        _AGGREGATION_FUNCTIONS.get(aggregation, aggregation.upper()),
        exp.column(metric_column),
    )
    for index, condition in enumerate(raw_conditions):
        if not isinstance(condition, dict):
            errors.append(f"HAVING condition {index} must be a typed mapping.")
            continue
        if condition.get("column") != metric_column:
            errors.append(f"HAVING condition {index} must target metric_column.")
            continue
        operator = condition.get("operator")
        if operator not in _COMPARISON_EXPRESSIONS:
            errors.append(f"Unsupported HAVING operator: {operator!r}.")
            continue
        value = condition.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"HAVING condition {index} requires a numeric value.")
            continue
        if not math.isfinite(float(value)):
            errors.append(f"HAVING condition {index} requires a finite value.")
            continue
        expressions.append(
            _COMPARISON_EXPRESSIONS[operator](
                this=aggregate_expression.copy(),
                expression=exp.Literal.number(str(value)),
            )
        )
    return expressions


def _compile_order_by(
    value: Any,
    aggregation: str | None,
    group_by: list[str],
    selected_columns: list[str],
    allowed_raw_columns: set[str],
    errors: list[str],
) -> exp.Expression | None:
    if value is None:
        return None
    if not isinstance(value, str):
        errors.append("order_by must be a bounded column-and-direction string.")
        return None
    match = _ORDER_BY.fullmatch(value.strip())
    if not match:
        errors.append("order_by contains an unsupported expression.")
        return None
    name = match.group("name")
    direction = (match.group("direction") or "ASC").upper()
    if aggregation:
        if not group_by:
            errors.append("A scalar aggregate cannot specify order_by.")
            return None
        allowed = {*group_by, f"{aggregation}_value"}
    else:
        allowed = allowed_raw_columns.intersection(selected_columns)
    if name not in allowed:
        errors.append(f"order_by target is not allowed for this plan: {name}.")
        return None
    column = exp.column(name)
    return column.desc() if direction == "DESC" else column.asc()


def _validate_limit(
    value: Any, requirements: dict[str, Any], errors: list[str]
) -> int | None:
    safety = requirements.get("safety")
    if not isinstance(safety, dict):
        errors.append("requirements_context.safety must be a mapping.")
        return None
    require_limit = safety.get("require_limit")
    max_limit = safety.get("max_limit")
    if type(require_limit) is not bool or type(max_limit) is not int or max_limit < 1:
        errors.append("SQL limit safety settings are invalid.")
        return None
    if value is None and not require_limit:
        return None
    if type(value) is not int:
        errors.append("limit must be an integer.")
        return None
    if value < 1 or value > max_limit:
        errors.append(f"limit must be between 1 and {max_limit}.")
        return None
    return value
