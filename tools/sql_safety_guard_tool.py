from typing import Any

from sqlglot import ErrorLevel, exp, parse
from sqlglot.errors import SqlglotError
from sqlglot.optimizer.scope import Scope, traverse_scope


def validate_sql_safety_tool(
    sql_query: str | None,
    requirements_context: dict[str, Any],
    permitted_table: str | None,
) -> dict[str, Any]:
    """Validate the project's deliberately narrow read-only BigQuery policy."""
    if not sql_query or not sql_query.strip():
        return _result(["SQL query is empty."])

    if not isinstance(permitted_table, str) or not permitted_table.strip():
        return _result(["Permitted logical table is not configured."])

    safety_config = requirements_context.get("safety", {})
    if not isinstance(safety_config, dict):
        return _result(["SQL safety configuration is invalid."])

    require_limit = safety_config.get("require_limit", True)
    max_limit = safety_config.get("max_limit")

    if not isinstance(require_limit, bool):
        return _result(["SQL safety require_limit setting must be a boolean."])

    if type(max_limit) is not int or max_limit < 1:
        return _result(["SQL safety max_limit setting must be a positive integer."])

    try:
        parsed_statements = parse(
            sql_query,
            read="bigquery",
            error_level=ErrorLevel.RAISE,
        )
    except SqlglotError:
        return _result(["SQL could not be parsed as valid BigQuery SQL."])
    except Exception:
        return _result(["SQL validation failed while parsing the query."])

    if len(parsed_statements) != 1:
        return _result(["Exactly one SQL statement is allowed."])

    statement = parsed_statements[0]
    if statement is None:
        return _result(["SQL could not be parsed as valid BigQuery SQL."])

    if not isinstance(statement, exp.Select):
        return _result(["Only a single read-only SELECT query is supported."])

    errors: list[str] = []

    if statement.find(exp.SetOperation):
        errors.append("Set operations such as UNION, INTERSECT, and EXCEPT are not supported.")

    if statement.find(exp.Into):
        errors.append("SELECT INTO is not supported.")

    if any(not select.expressions for select in statement.find_all(exp.Select)):
        errors.append("Every SELECT query must include at least one projection.")

    errors.extend(
        _validate_query_sources(
            statement=statement,
            permitted_table=permitted_table.strip(),
        )
    )

    limit_error = _validate_outer_limit(
        statement=statement,
        require_limit=require_limit,
        max_limit=max_limit,
    )
    if limit_error:
        errors.append(limit_error)

    return _result(_deduplicate(errors))


def _validate_query_sources(
    statement: exp.Select,
    permitted_table: str,
) -> list[str]:
    errors: list[str] = []
    found_permitted_table = False

    try:
        scopes = traverse_scope(statement)
    except SqlglotError:
        return ["SQL query sources could not be resolved safely."]
    except Exception:
        return ["SQL validation failed while resolving query sources."]

    if not scopes:
        return ["SQL query sources could not be resolved safely."]

    for scope in scopes:
        if not isinstance(scope.expression, exp.Select):
            errors.append("Table-valued and external query sources are not supported.")
            continue

        try:
            selected_sources = scope.selected_sources.values()
        except SqlglotError:
            errors.append("SQL query sources could not be resolved safely.")
            continue

        for _, source in selected_sources:
            if isinstance(source, Scope):
                continue

            if not isinstance(source, exp.Table):
                errors.append("Unsupported SQL query source.")
                continue

            if not isinstance(source.this, exp.Identifier):
                errors.append("Table-valued and external query sources are not supported.")
                continue

            table_name = source.name

            if "*" in table_name:
                errors.append("Wildcard table references are not supported.")
                continue

            if source.catalog or source.db:
                errors.append(
                    "Qualified physical table references are not allowed: "
                    f"{source.sql(dialect='bigquery')}."
                )
                continue

            if table_name != permitted_table:
                errors.append(f"Unauthorized table reference: {table_name or '<unknown>'}.")
                continue

            found_permitted_table = True

    if not found_permitted_table:
        errors.append(
            f"Query must read from the permitted logical table '{permitted_table}'."
        )

    return _deduplicate(errors)


def _validate_outer_limit(
    statement: exp.Select,
    require_limit: bool,
    max_limit: int,
) -> str | None:
    limit = statement.args.get("limit")

    if limit is None:
        if require_limit:
            return "SQL query must include an outermost LIMIT clause."
        return None

    limit_expression = limit.args.get("expression")

    if not isinstance(limit_expression, exp.Literal) or limit_expression.is_string:
        return "Outermost LIMIT must be a literal integer."

    raw_limit = str(limit_expression.this)
    if not raw_limit.isascii() or not raw_limit.isdigit():
        return "Outermost LIMIT must be a literal integer."

    limit_value = int(raw_limit)
    if limit_value < 1 or limit_value > max_limit:
        return f"Outermost LIMIT must be between 1 and {max_limit}."

    return None


def _deduplicate(errors: list[str]) -> list[str]:
    return list(dict.fromkeys(errors))


def _result(errors: list[str]) -> dict[str, Any]:
    return {
        "is_safe": not errors,
        "errors": errors,
    }
