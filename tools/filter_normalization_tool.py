from typing import Any

from agent.state import QueryFilter


FilterItem = QueryFilter

def deduplicate_filters(filters: list[FilterItem]) -> list[FilterItem]:
    unique_filters: list[FilterItem] = []
    seen = set()

    for filter_item in filters:
        column = filter_item.get("column")
        operator = filter_item.get("operator")
        value = filter_item.get("value")

        key = (column, operator, str(value))

        if key in seen:
            continue

        seen.add(key)
        unique_filters.append(filter_item)

    return unique_filters


def is_empty_filter_value(value: Any) -> bool:
    if value is None:
        return True

    if isinstance(value, str):
        return value.strip() == ""

    if isinstance(value, (list, tuple, set)):
        return len(value) == 0

    return False


def iter_filter_values(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple, set)):
        return [
            item
            for item in value
            if not is_empty_filter_value(item)
        ]

    if is_empty_filter_value(value):
        return []

    return [value]


def filter_value_key(value: Any) -> str:
    if isinstance(value, str):
        return value.strip().lower()

    return str(value).strip().lower()


def normalize_multi_value_equality_filters(
    filters: list[FilterItem],
) -> tuple[list[FilterItem], set[str]]:
    equality_values_by_column: dict[str, list[Any]] = {}
    equality_value_keys_by_column: dict[str, set[str]] = {}

    for filter_item in filters:
        column = filter_item.get("column")
        operator = str(filter_item.get("operator") or "").strip().upper()

        if not column or operator != "=":
            continue

        values = iter_filter_values(filter_item.get("value"))

        if not values:
            continue

        if column not in equality_values_by_column:
            equality_values_by_column[column] = []
            equality_value_keys_by_column[column] = set()

        for value in values:
            value_key = filter_value_key(value)

            if value_key in equality_value_keys_by_column[column]:
                continue

            equality_value_keys_by_column[column].add(value_key)
            equality_values_by_column[column].append(value)

    multi_value_filter_columns = {
        column
        for column, values in equality_values_by_column.items()
        if len(values) > 1
    }

    emitted_equality_columns: set[str] = set()
    normalized_filters: list[FilterItem] = []

    for filter_item in filters:
        column = filter_item.get("column")
        operator = str(filter_item.get("operator") or "").strip().upper()

        if not column or operator != "=":
            normalized_filters.append(filter_item)
            continue

        if column in emitted_equality_columns:
            continue

        emitted_equality_columns.add(column)

        values = equality_values_by_column.get(column, [])

        if not values:
            continue

        if column in multi_value_filter_columns:
            normalized_filters.append(
                {
                    "column": column,
                    "operator": "IN",
                    "value": values,
                }
            )
        else:
            normalized_filters.append(
                {
                    "column": column,
                    "operator": "=",
                    "value": values[0],
                }
            )

    return normalized_filters, multi_value_filter_columns