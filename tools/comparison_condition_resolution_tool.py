import re
from typing import Any


_COMPARISON_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"\b(?:greater\s+than|more\s+than|above|over)\s+(-?\d+(?:\.\d+)?)\b",
            flags=re.IGNORECASE,
        ),
        ">",
    ),
    (
        re.compile(
            r"\b(?:less\s+than|lower\s+than|below|under)\s+(-?\d+(?:\.\d+)?)\b",
            flags=re.IGNORECASE,
        ),
        "<",
    ),
    (
        re.compile(
            r"\b(?:at\s+least|greater\s+than\s+or\s+equal\s+to|greater\s+or\s+equal\s+to|not\s+less\s+than)\s+(-?\d+(?:\.\d+)?)\b",
            flags=re.IGNORECASE,
        ),
        ">=",
    ),
    (
        re.compile(
            r"\b(?:at\s+most|less\s+than\s+or\s+equal\s+to|less\s+or\s+equal\s+to|not\s+greater\s+than)\s+(-?\d+(?:\.\d+)?)\b",
            flags=re.IGNORECASE,
        ),
        "<=",
    ),
]


_DIMENSION_OUTPUT_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "district": [
        re.compile(r"\bshow\s+districts?\s+where\b", flags=re.IGNORECASE),
        re.compile(r"\blist\s+districts?\s+where\b", flags=re.IGNORECASE),
        re.compile(r"\bwhich\s+districts?\b", flags=re.IGNORECASE),
    ],
    "city": [
        re.compile(r"\bshow\s+cities\s+where\b", flags=re.IGNORECASE),
        re.compile(r"\blist\s+cities\s+where\b", flags=re.IGNORECASE),
        re.compile(r"\bwhich\s+cities\b", flags=re.IGNORECASE),
    ],
    "sensor_id": [
        re.compile(r"\bshow\s+sensors?\s+where\b", flags=re.IGNORECASE),
        re.compile(r"\blist\s+sensors?\s+where\b", flags=re.IGNORECASE),
        re.compile(r"\bwhich\s+sensors?\b", flags=re.IGNORECASE),
    ],
    "type": [
        re.compile(r"\bshow\s+(?:measurement\s+)?types?\s+where\b", flags=re.IGNORECASE),
        re.compile(r"\blist\s+(?:measurement\s+)?types?\s+where\b", flags=re.IGNORECASE),
        re.compile(r"\bwhich\s+(?:measurement\s+)?types?\b", flags=re.IGNORECASE),
    ],
}


def resolve_comparison_conditions_from_text(
    request_text: str,
    metric_column: str | None,
    aggregation: str | None,
    available_columns: set[str] | list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """
    Resolve numeric comparison conditions from the request text.

    This tool is intentionally deterministic and conservative.

    It only handles explicit numeric threshold phrases such as:
    - greater than 2
    - less than 2
    - above 2
    - below 2
    - at least 2
    - at most 2
    - negative values

    It does not infer conditions from dates, limits, rankings, or comparisons
    between locations.
    """

    normalized_request_text = _normalize_request_text(request_text)

    if not normalized_request_text or not metric_column:
        return _empty_result()

    available_column_set = _normalize_available_columns(available_columns)

    if available_column_set is not None and metric_column not in available_column_set:
        return _empty_result()

    condition = _extract_first_numeric_condition(
        request_text=normalized_request_text,
        metric_column=metric_column,
    )

    if condition is None:
        condition = _extract_negative_condition(
            request_text=normalized_request_text,
            metric_column=metric_column,
        )

    requested_output_columns = _detect_requested_output_columns(
        request_text=normalized_request_text,
        available_columns=available_column_set,
    )

    if condition is None:
        return {
            "where_conditions": [],
            "having_conditions": [],
            "requested_output_columns": requested_output_columns,
            "distinct": False,
        }

    has_aggregation = isinstance(aggregation, str) and bool(aggregation.strip())

    if has_aggregation:
        return {
            "where_conditions": [],
            "having_conditions": [condition],
            "requested_output_columns": requested_output_columns,
            "distinct": False,
        }

    return {
        "where_conditions": [condition],
        "having_conditions": [],
        "requested_output_columns": requested_output_columns,
        "distinct": bool(requested_output_columns),
    }


def _empty_result() -> dict[str, Any]:
    return {
        "where_conditions": [],
        "having_conditions": [],
        "requested_output_columns": [],
        "distinct": False,
    }


def _normalize_request_text(request_text: str) -> str:
    return " ".join(str(request_text or "").strip().split())


def _normalize_available_columns(
    available_columns: set[str] | list[str] | tuple[str, ...] | None,
) -> set[str] | None:
    if available_columns is None:
        return None

    return {
        str(column)
        for column in available_columns
        if isinstance(column, str) and column.strip()
    }


def _parse_number(value: str) -> int | float:
    if "." in value:
        return float(value)

    return int(value)


def _extract_first_numeric_condition(
    request_text: str,
    metric_column: str,
) -> dict[str, Any] | None:
    for pattern, operator in _COMPARISON_PATTERNS:
        match = pattern.search(request_text)

        if not match:
            continue

        raw_value = match.group(1)

        return {
            "column": metric_column,
            "operator": operator,
            "value": _parse_number(raw_value),
            "source": match.group(0),
        }

    return None


def _extract_negative_condition(
    request_text: str,
    metric_column: str,
) -> dict[str, Any] | None:
    if not re.search(r"\bnegative\b", request_text, flags=re.IGNORECASE):
        return None

    if not re.search(
        r"\b(?:flow|value|values|measurement|measurements|sensor\s+value|sensor\s+values)\b",
        request_text,
        flags=re.IGNORECASE,
    ):
        return None

    return {
        "column": metric_column,
        "operator": "<",
        "value": 0,
        "source": "negative",
    }


def _detect_requested_output_columns(
    request_text: str,
    available_columns: set[str] | None,
) -> list[str]:
    requested_columns: list[str] = []

    for column, patterns in _DIMENSION_OUTPUT_PATTERNS.items():
        if available_columns is not None and column not in available_columns:
            continue

        if any(pattern.search(request_text) for pattern in patterns):
            requested_columns.append(column)

    return requested_columns