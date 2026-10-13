from typing import Any


def resolve_schema_table_context(
    mapped_table_name: object,
    previous_table_name: object,
    schema_context: object,
    requirements_context: object,
) -> tuple[str | None, dict[str, Any], set[str]]:
    """
    Resolves the active table name, requirement defaults, and available columns.

    Rules:
    - Prefer mapped table name.
    - Fall back to previous state table name.
    - Fall back to requirements defaults table.
    - Resolve available columns from schema_context.tables[table_name].columns.
    """
    defaults = _extract_defaults(requirements_context)

    table_name = (
        _to_string_or_none(mapped_table_name)
        or _to_string_or_none(previous_table_name)
        or _to_string_or_none(defaults.get("table"))
    )

    table_data = _extract_table_data(
        schema_context=schema_context,
        table_name=table_name,
    )

    columns = table_data.get("columns", {})
    available_columns = set(columns.keys()) if isinstance(columns, dict) else set()

    return table_name, defaults, available_columns


def _extract_defaults(requirements_context: object) -> dict[str, Any]:
    """
    Extracts defaults from requirements context.
    """
    if not isinstance(requirements_context, dict):
        return {}

    defaults = requirements_context.get("defaults", {})

    if not isinstance(defaults, dict):
        return {}

    return defaults


def _extract_table_data(
    schema_context: object,
    table_name: str | None,
) -> dict[str, Any]:
    """
    Extracts table data from schema context.
    """
    if not table_name:
        return {}

    if not isinstance(schema_context, dict):
        return {}

    tables = schema_context.get("tables", {})

    if not isinstance(tables, dict):
        return {}

    table_data = tables.get(table_name, {})

    if not isinstance(table_data, dict):
        return {}

    return table_data


def _to_string_or_none(value: object) -> str | None:
    """
    Converts a raw value into a non-empty string, otherwise returns None.
    """
    if isinstance(value, str) and value.strip():
        return value

    return None