import re
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

from agent.state import TimeRange


TimeRangeUnit = Literal["SECOND", "MINUTE", "HOUR", "DAY"]


_UNIT_ALIASES: dict[str, TimeRangeUnit] = {
    "s": "SECOND",
    "sec": "SECOND",
    "secs": "SECOND",
    "second": "SECOND",
    "seconds": "SECOND",
    "min": "MINUTE",
    "mins": "MINUTE",
    "minute": "MINUTE",
    "minutes": "MINUTE",
    "h": "HOUR",
    "hr": "HOUR",
    "hrs": "HOUR",
    "hour": "HOUR",
    "hours": "HOUR",
    "d": "DAY",
    "day": "DAY",
    "days": "DAY",
    "w": "DAY",
    "week": "DAY",
    "weeks": "DAY",
}


_VALID_TIME_UNITS: set[str] = {"SECOND", "MINUTE", "HOUR", "DAY"}

_DATETIME_VALUE_PATTERN = (
    r"\d{4}-\d{1,2}-\d{1,2}"
    r"(?:[T\s]+\d{1,2}:\d{2}"
    r"(?::\d{2})?"
    r"(?:\s*(?:Z|[+-]\d{2}(?::?\d{2})?))?"
    r")?"
)

_PARSED_DATETIME_PATTERN = re.compile(
    r"^(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})"
    r"(?:[T\s]+(?P<hour>\d{1,2}):(?P<minute>\d{2})"
    r"(?::(?P<second>\d{2}))?"
    r"(?:\s*(?P<timezone>Z|[+-]\d{2}(?::?\d{2})?))?"
    r")?$",
    flags=re.IGNORECASE,
)

_SAFE_TIMESTAMP_LITERAL_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}"
    r"(?:Z|[+-]\d{2}(?::?\d{2})?)?$",
    flags=re.IGNORECASE,
)


def normalize_time_range_tool(text: str | None) -> Optional[TimeRange]:
    """
    Normalize a natural-language time phrase into a canonical TimeRange.

    This function does not generate SQL.
    It only creates a deterministic structure that the SQL generator can later
    compile into BigQuery SQL.
    """

    if not text:
        return None

    absolute_range = _parse_absolute_range(text)

    if absolute_range:
        return absolute_range

    normalized_text = _normalize_text(text)

    named_range = _parse_named_range(normalized_text)

    if named_range:
        return named_range

    relative_range = _parse_relative_range(normalized_text)

    if relative_range:
        return relative_range

    return None


def normalize_bare_date_tool(text: str | None) -> Optional[TimeRange]:
    """Use the existing parser only when the whole answer is an ISO-like date."""
    if not text or re.fullmatch(r"\s*\d{4}-\d{1,2}-\d{1,2}\s*", text) is None:
        return None
    return normalize_time_range_tool(text)


def extract_single_explicit_absolute_date_tool(text: str | None) -> dict:
    """Extract one unquoted, non-negated ISO-like date and validate it.

    This is deliberately narrower than ``normalize_time_range_tool``: it does
    not interpret relative or named dates and it refuses multiple candidates.
    """
    if not text:
        return {"status": "not_found", "phrase": None, "parsed": None}

    unquoted_text = re.sub(r"(['\"]).*?\1", " ", text)
    quoted_candidates = re.findall(r"\b\d{4}-\d{1,2}-\d{1,2}\b", text)
    candidates = list(re.finditer(r"\b\d{4}-\d{1,2}-\d{1,2}\b", unquoted_text))

    if not candidates:
        status = "rejected_quoted" if quoted_candidates else "not_found"
        return {"status": status, "phrase": None, "parsed": None}
    if len(candidates) != 1:
        return {"status": "ambiguous_multiple", "phrase": None, "parsed": None}

    match = candidates[0]
    prefix = unquoted_text[max(0, match.start() - 48):match.start()].lower()
    if re.search(
        r"(?:\bnot|\bwithout|\bexcept|\bexclude|\bexcluding|\bignore|\bavoid|"
        r"\bdo\s+not\s+use|\bdon't\s+use)\s+(?:on\s+)?(?:the\s+date\s+)?$",
        prefix,
    ):
        return {"status": "rejected_negated", "phrase": None, "parsed": None}

    phrase = match.group(0)
    parsed = normalize_bare_date_tool(phrase)
    if parsed is None:
        return {"status": "invalid", "phrase": phrase, "parsed": None}
    return {"status": "applied", "phrase": phrase, "parsed": parsed}


def build_bigquery_time_predicate_tool(
    time_range: TimeRange | None,
    time_column: str | None,
) -> str | None:
    """
    Build a deterministic BigQuery Standard SQL time predicate from TimeRange.

    This function is used after time phrases have already been normalized.
    It prevents the LLM from generating incorrect timestamp predicates such as:
        timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
    for named ranges like yesterday.
    """

    if not time_range or not time_column:
        return None

    if not _is_safe_column_name(time_column):
        return None

    kind = time_range.get("kind")

    if kind == "relative":
        return _build_relative_bigquery_predicate(
            time_range=time_range,
            time_column=time_column,
        )

    if kind == "named":
        return _build_named_bigquery_predicate(
            time_range=time_range,
            time_column=time_column,
        )

    if kind == "absolute":
        return _build_absolute_bigquery_predicate(
            time_range=time_range,
            time_column=time_column,
        )

    return _build_backward_compatible_bigquery_predicate(
        time_range=time_range,
        time_column=time_column,
    )


def _normalize_text(text: str) -> str:
    normalized_text = text.lower().strip()

    normalized_text = normalized_text.replace("_", " ")
    normalized_text = normalized_text.replace("-", " ")

    return " ".join(normalized_text.split())


def _normalize_absolute_text(text: str) -> str:
    normalized_text = text.lower().strip()

    normalized_text = normalized_text.replace("_", " ")

    return " ".join(normalized_text.split())


def _parse_absolute_range(text: str) -> Optional[TimeRange]:
    normalized_text = _normalize_absolute_text(text)

    from_to_match = re.search(
        rf"\bfrom\s+({_DATETIME_VALUE_PATTERN})\s+"
        rf"(?:to|until|through|till)\s+({_DATETIME_VALUE_PATTERN})\b",
        normalized_text,
        flags=re.IGNORECASE,
    )

    if from_to_match:
        return _build_absolute_time_range(
            start_value=from_to_match.group(1),
            end_value=from_to_match.group(2),
            end_date_is_inclusive=True,
        )

    between_match = re.search(
        rf"\bbetween\s+({_DATETIME_VALUE_PATTERN})\s+"
        rf"and\s+({_DATETIME_VALUE_PATTERN})\b",
        normalized_text,
        flags=re.IGNORECASE,
    )

    if between_match:
        return _build_absolute_time_range(
            start_value=between_match.group(1),
            end_value=between_match.group(2),
            end_date_is_inclusive=True,
        )

    bare_range_match = re.search(
        rf"\b({_DATETIME_VALUE_PATTERN})\s+"
        rf"(?:to|until|through|till)\s+({_DATETIME_VALUE_PATTERN})\b",
        normalized_text,
        flags=re.IGNORECASE,
    )

    if bare_range_match:
        return _build_absolute_time_range(
            start_value=bare_range_match.group(1),
            end_value=bare_range_match.group(2),
            end_date_is_inclusive=True,
        )

    on_match = re.search(
        rf"\bon\s+({_DATETIME_VALUE_PATTERN})\b",
        normalized_text,
        flags=re.IGNORECASE,
    )

    if on_match:
        return _build_absolute_time_range(
            start_value=on_match.group(1),
            end_value=on_match.group(1),
            end_date_is_inclusive=True,
        )

    bare_single_date_match = re.fullmatch(
        r"\d{4}-\d{1,2}-\d{1,2}",
        normalized_text,
        flags=re.IGNORECASE,
    )

    if bare_single_date_match:
        return _build_absolute_time_range(
            start_value=bare_single_date_match.group(0),
            end_value=bare_single_date_match.group(0),
            end_date_is_inclusive=True,
        )

    after_match = re.search(
        rf"\b(?:after|since)\s+({_DATETIME_VALUE_PATTERN})\b",
        normalized_text,
        flags=re.IGNORECASE,
    )

    if after_match:
        return _build_absolute_time_range(
            start_value=after_match.group(1),
            end_value=None,
            end_date_is_inclusive=False,
        )

    single_from_match = re.search(
        rf"\bfrom\s+({_DATETIME_VALUE_PATTERN})\b",
        normalized_text,
        flags=re.IGNORECASE,
    )

    if single_from_match:
        return _build_absolute_time_range(
            start_value=single_from_match.group(1),
            end_value=None,
            end_date_is_inclusive=False,
        )

    before_match = re.search(
        rf"\bbefore\s+({_DATETIME_VALUE_PATTERN})\b",
        normalized_text,
        flags=re.IGNORECASE,
    )

    if before_match:
        return _build_absolute_time_range(
            start_value=None,
            end_value=before_match.group(1),
            end_date_is_inclusive=False,
        )

    return None


def _build_absolute_time_range(
    start_value: str | None,
    end_value: str | None,
    end_date_is_inclusive: bool,
) -> Optional[TimeRange]:
    start_result = _parse_datetime_value(start_value) if start_value else None
    end_result = _parse_datetime_value(end_value) if end_value else None

    if start_value and start_result is None:
        return None

    if end_value and end_result is None:
        return None

    start_datetime = start_result[0] if start_result else None
    end_datetime = end_result[0] if end_result else None
    end_has_time = end_result[1] if end_result else False

    if end_datetime is not None and end_date_is_inclusive and not end_has_time:
        end_datetime = end_datetime + timedelta(days=1)

    if (
        start_datetime is not None
        and end_datetime is not None
        and start_datetime >= end_datetime
    ):
        return None

    time_range: TimeRange = {
        "kind": "absolute",
        "relative": None,
    }

    if start_datetime is not None:
        time_range["start"] = _format_datetime_for_bigquery(start_datetime)

    if end_datetime is not None:
        time_range["end"] = _format_datetime_for_bigquery(end_datetime)

    return time_range


def _parse_datetime_value(value: str | None) -> tuple[datetime, bool] | None:
    if not value:
        return None

    cleaned_value = value.strip()

    match = _PARSED_DATETIME_PATTERN.fullmatch(cleaned_value)

    if not match:
        return None

    try:
        year = int(match.group("year"))
        month = int(match.group("month"))
        day = int(match.group("day"))

        hour_text = match.group("hour")
        minute_text = match.group("minute")
        second_text = match.group("second")
        timezone_text = match.group("timezone")

        has_time = hour_text is not None

        hour = int(hour_text) if hour_text is not None else 0
        minute = int(minute_text) if minute_text is not None else 0
        second = int(second_text) if second_text is not None else 0

        parsed_timezone = _parse_timezone_value(timezone_text)

        if parsed_timezone is None:
            return None

        parsed_datetime = datetime(
            year=year,
            month=month,
            day=day,
            hour=hour,
            minute=minute,
            second=second,
            tzinfo=parsed_timezone,
        )

        return parsed_datetime.astimezone(timezone.utc), has_time

    except ValueError:
        return None


def _parse_timezone_value(value: str | None) -> timezone | None:
    if value is None or value.strip() == "":
        return timezone.utc

    cleaned_value = value.strip().upper()

    if cleaned_value == "Z":
        return timezone.utc

    timezone_match = re.fullmatch(
        r"(?P<sign>[+-])(?P<hour>\d{2})(?::?(?P<minute>\d{2}))?",
        cleaned_value,
    )

    if not timezone_match:
        return None

    sign = 1 if timezone_match.group("sign") == "+" else -1
    hour = int(timezone_match.group("hour"))
    minute_text = timezone_match.group("minute")
    minute = int(minute_text) if minute_text is not None else 0

    if hour > 23 or minute > 59:
        return None

    offset = timedelta(hours=hour, minutes=minute)

    return timezone(sign * offset)


def _format_datetime_for_bigquery(value: datetime) -> str:
    utc_value = value.astimezone(timezone.utc)

    return utc_value.strftime("%Y-%m-%d %H:%M:%S+00")


def _parse_named_range(text: str) -> Optional[TimeRange]:
    if text == "today" or " today" in f" {text} ":
        return {
            "kind": "named",
            "name": "today",
            "relative": "today",
        }

    if text == "yesterday" or " yesterday" in f" {text} ":
        return {
            "kind": "named",
            "name": "yesterday",
            "relative": "yesterday",
        }

    if "last week" in text:
        return _build_relative_time_range(
            amount=7,
            unit="DAY",
            relative="last_7_days",
        )

    if "last month" in text:
        return _build_relative_time_range(
            amount=30,
            unit="DAY",
            relative="last_30_days",
        )

    return None


def _parse_relative_range(text: str) -> Optional[TimeRange]:
    spaced_match = re.search(
        r"\b(?:last|past|previous)\s+(\d+)\s*"
        r"(seconds?|secs?|sec|s|minutes?|mins?|min|hours?|hrs?|hr|h|days?|d|weeks?|w)\b",
        text,
    )

    if spaced_match:
        amount = int(spaced_match.group(1))
        raw_unit = spaced_match.group(2)

        return _build_relative_time_range_from_raw_unit(
            amount=amount,
            raw_unit=raw_unit,
        )

    compact_match = re.search(
        r"\b(?:last|past|previous)\s+(\d+)(s|sec|secs|min|mins|h|hr|hrs|d|w)\b",
        text,
    )

    if compact_match:
        amount = int(compact_match.group(1))
        raw_unit = compact_match.group(2)

        return _build_relative_time_range_from_raw_unit(
            amount=amount,
            raw_unit=raw_unit,
        )

    return None


def _build_relative_time_range_from_raw_unit(
    amount: int,
    raw_unit: str,
) -> Optional[TimeRange]:
    if amount <= 0:
        return None

    normalized_unit = _UNIT_ALIASES.get(raw_unit)

    if normalized_unit is None:
        return None

    relative_unit_name = _get_relative_unit_name(raw_unit)

    if relative_unit_name == "weeks":
        amount = amount * 7
        normalized_unit = "DAY"

    relative = f"last_{amount}_{_unit_to_relative_suffix(normalized_unit)}"

    return _build_relative_time_range(
        amount=amount,
        unit=normalized_unit,
        relative=relative,
    )


def _build_relative_time_range(
    amount: int,
    unit: TimeRangeUnit,
    relative: str,
) -> TimeRange:
    return {
        "kind": "relative",
        "amount": amount,
        "unit": unit,
        "relative": relative,
    }


def _get_relative_unit_name(raw_unit: str) -> str:
    raw_unit = raw_unit.lower().strip()

    if raw_unit in {"w", "week", "weeks"}:
        return "weeks"

    if raw_unit in {"d", "day", "days"}:
        return "days"

    if raw_unit in {"h", "hr", "hrs", "hour", "hours"}:
        return "hours"

    if raw_unit in {"min", "mins", "minute", "minutes"}:
        return "minutes"

    if raw_unit in {"s", "sec", "secs", "second", "seconds"}:
        return "seconds"

    return raw_unit


def _unit_to_relative_suffix(unit: TimeRangeUnit) -> str:
    if unit == "SECOND":
        return "seconds"

    if unit == "MINUTE":
        return "minutes"

    if unit == "HOUR":
        return "hours"

    if unit == "DAY":
        return "days"

    return unit.lower()


def _build_relative_bigquery_predicate(
    time_range: TimeRange,
    time_column: str,
) -> str | None:
    amount = time_range.get("amount")
    unit = time_range.get("unit")

    if not isinstance(amount, int):
        return None

    if unit not in _VALID_TIME_UNITS:
        return None

    return (
        f"{time_column} >= "
        f"TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {amount} {unit})"
    )


def _build_named_bigquery_predicate(
    time_range: TimeRange,
    time_column: str,
) -> str | None:
    name = time_range.get("name")

    if name == "today":
        return _build_today_bigquery_predicate(time_column)

    if name == "yesterday":
        return _build_yesterday_bigquery_predicate(time_column)

    return None


def _build_absolute_bigquery_predicate(
    time_range: TimeRange,
    time_column: str,
) -> str | None:
    start = time_range.get("start")
    end = time_range.get("end")

    conditions: list[str] = []

    if isinstance(start, str) and start.strip():
        start_value = start.strip()

        if not _is_safe_timestamp_literal(start_value):
            return None

        conditions.append(f"{time_column} >= TIMESTAMP('{start_value}')")

    if isinstance(end, str) and end.strip():
        end_value = end.strip()

        if not _is_safe_timestamp_literal(end_value):
            return None

        conditions.append(f"{time_column} < TIMESTAMP('{end_value}')")

    if not conditions:
        return None

    return " AND ".join(conditions)


def _build_backward_compatible_bigquery_predicate(
    time_range: TimeRange,
    time_column: str,
) -> str | None:
    relative = time_range.get("relative")

    if not isinstance(relative, str):
        return None

    if relative == "today":
        return _build_today_bigquery_predicate(time_column)

    if relative == "yesterday":
        return _build_yesterday_bigquery_predicate(time_column)

    relative_match = re.fullmatch(
        r"last_(\d+)_(seconds|minutes|hours|days)",
        relative,
    )

    if not relative_match:
        return None

    amount = int(relative_match.group(1))
    unit_suffix = relative_match.group(2)

    unit = _relative_suffix_to_bigquery_unit(unit_suffix)

    if unit is None:
        return None

    return (
        f"{time_column} >= "
        f"TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {amount} {unit})"
    )


def _build_today_bigquery_predicate(time_column: str) -> str:
    return (
        f"{time_column} >= TIMESTAMP_TRUNC(CURRENT_TIMESTAMP(), DAY) "
        f"AND {time_column} < "
        f"TIMESTAMP_ADD(TIMESTAMP_TRUNC(CURRENT_TIMESTAMP(), DAY), INTERVAL 1 DAY)"
    )


def _build_yesterday_bigquery_predicate(time_column: str) -> str:
    return (
        f"{time_column} >= "
        f"TIMESTAMP_SUB(TIMESTAMP_TRUNC(CURRENT_TIMESTAMP(), DAY), INTERVAL 1 DAY) "
        f"AND {time_column} < TIMESTAMP_TRUNC(CURRENT_TIMESTAMP(), DAY)"
    )


def _relative_suffix_to_bigquery_unit(unit_suffix: str) -> TimeRangeUnit | None:
    if unit_suffix == "seconds":
        return "SECOND"

    if unit_suffix == "minutes":
        return "MINUTE"

    if unit_suffix == "hours":
        return "HOUR"

    if unit_suffix == "days":
        return "DAY"

    return None


def _is_safe_column_name(column_name: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", column_name))


def _is_safe_timestamp_literal(timestamp_value: str) -> bool:
    return bool(_SAFE_TIMESTAMP_LITERAL_PATTERN.fullmatch(timestamp_value))


def enforce_bigquery_time_predicate_tool(
    sql_query: str,
    time_range: TimeRange | None,
    time_column: str | None,
) -> str:
    """
    Enforce the deterministic BigQuery time predicate inside a generated SQL query.

    This function removes existing time conditions from the WHERE clause and
    appends the deterministic predicate built from the canonical TimeRange.
    """

    deterministic_predicate = build_bigquery_time_predicate_tool(
        time_range=time_range,
        time_column=time_column,
    )

    if not sql_query or not deterministic_predicate or not time_column:
        return sql_query

    split_sql = _split_sql_around_where_clause(sql_query)

    if split_sql is None:
        return _insert_where_clause(
            sql_query=sql_query,
            predicate=deterministic_predicate,
        )

    prefix, where_clause, suffix = split_sql

    non_time_conditions = _get_non_time_conditions(
        where_clause=where_clause,
        time_column=time_column,
    )

    updated_conditions = [
        *non_time_conditions,
        deterministic_predicate,
    ]

    updated_where_clause = " AND ".join(updated_conditions)

    if suffix:
        return f"{prefix} WHERE {updated_where_clause} {suffix}".strip()

    return f"{prefix} WHERE {updated_where_clause}".strip()


def _split_sql_around_where_clause(
    sql_query: str,
) -> tuple[str, str, str] | None:
    where_match = re.search(
        r"\bWHERE\b",
        sql_query,
        flags=re.IGNORECASE,
    )

    if not where_match:
        return None

    prefix = sql_query[: where_match.start()].strip()
    after_where = sql_query[where_match.end() :]

    next_clause_match = re.search(
        r"\b(GROUP\s+BY|ORDER\s+BY|LIMIT)\b",
        after_where,
        flags=re.IGNORECASE,
    )

    if next_clause_match:
        where_clause = after_where[: next_clause_match.start()].strip()
        suffix = after_where[next_clause_match.start() :].strip()

        return prefix, where_clause, suffix

    where_clause = after_where.strip()

    return prefix, where_clause, ""


def _insert_where_clause(
    sql_query: str,
    predicate: str,
) -> str:
    next_clause_match = re.search(
        r"\b(GROUP\s+BY|ORDER\s+BY|LIMIT)\b",
        sql_query,
        flags=re.IGNORECASE,
    )

    if not next_clause_match:
        return f"{sql_query.strip()} WHERE {predicate}"

    prefix = sql_query[: next_clause_match.start()].strip()
    suffix = sql_query[next_clause_match.start() :].strip()

    return f"{prefix} WHERE {predicate} {suffix}".strip()


def _get_non_time_conditions(
    where_clause: str,
    time_column: str,
) -> list[str]:
    conditions = re.split(
        r"\s+AND\s+",
        where_clause,
        flags=re.IGNORECASE,
    )

    non_time_conditions: list[str] = []

    for condition in conditions:
        cleaned_condition = condition.strip()

        if not cleaned_condition:
            continue

        if _is_time_condition(
            condition=cleaned_condition,
            time_column=time_column,
        ):
            continue

        non_time_conditions.append(cleaned_condition)

    return non_time_conditions


def _is_time_condition(
    condition: str,
    time_column: str,
) -> bool:
    normalized_condition = condition.upper()

    if re.search(
        rf"\b{re.escape(time_column)}\b",
        condition,
        flags=re.IGNORECASE,
    ):
        return True

    time_keywords = {
        "CURRENT_TIMESTAMP",
        "TIMESTAMP_SUB",
        "TIMESTAMP_ADD",
        "TIMESTAMP_TRUNC",
        "CURRENT_DATE",
        "TIMESTAMP(",
        "DATE(",
    }

    return any(keyword in normalized_condition for keyword in time_keywords)
