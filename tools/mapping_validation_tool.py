from typing import Any


def validate_mapping_tool(
    mapping: dict[str, Any],
    schema_context: dict[str, Any],
    semantic_context: dict[str, Any],
    requirements_context: dict[str, Any],
    public_schema_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    if public_schema_context is None:
        public_schema_context = {}

    table_name = mapping.get("table_name")
    available_tables = schema_context.get("tables", {})

    if not table_name:
        errors.append("Missing table_name.")
    elif table_name not in available_tables:
        errors.append(f"Unknown table_name: {table_name}")

    table_data = available_tables.get(table_name, {}) if table_name else {}
    columns = table_data.get("columns", {})
    available_columns = set(columns.keys())

    _validate_column(
        column=mapping.get("metric_column"),
        available_columns=available_columns,
        field_name="metric_column",
        errors=errors,
    )

    _validate_column(
        column=mapping.get("time_column"),
        available_columns=available_columns,
        field_name="time_column",
        errors=errors,
    )

    for column in mapping.get("selected_columns", []):
        _validate_column(
            column=column,
            available_columns=available_columns,
            field_name="selected_columns",
            errors=errors,
        )

    allowed_filter_columns = set(table_data.get("allowed_filter_columns", []))

    _validate_raw_metric_against_public_schema(
        raw_metric=mapping.get("raw_metric"),
        metric_column=mapping.get("metric_column"),
        public_schema_context=public_schema_context,
        errors=errors,
    )

    for filter_item in mapping.get("filters", []):
        column = filter_item.get("column")
        operator = str(filter_item.get("operator") or "").strip().upper()
        value = filter_item.get("value")

        if column not in allowed_filter_columns:
            errors.append(f"Filter column is not allowed: {column}")
            continue

        if column == "district":
            unsupported_values = _find_unsupported_filter_values(
                value=value,
                semantic_context=semantic_context,
                is_supported_fn=_is_supported_district,
            )

            for unsupported_value in unsupported_values:
                errors.append(f"Unsupported district value: {unsupported_value}")

        if column == "city":
            unsupported_values = _find_unsupported_filter_values(
                value=value,
                semantic_context=semantic_context,
                is_supported_fn=_is_supported_city,
            )

            for unsupported_value in unsupported_values:
                errors.append(f"Unsupported city value: {unsupported_value}")

        if column == "type":
            unsupported_values = _find_unsupported_filter_values(
                value=value,
                semantic_context=semantic_context,
                is_supported_fn=_is_supported_measurement_type,
            )

            for unsupported_value in unsupported_values:
                errors.append(f"Unsupported measurement type value: {unsupported_value}")

        if operator == "IN" and not isinstance(value, (list, tuple, set)):
            warnings.append(
                f"IN filter for column '{column}' should use a list value."
            )

    chart_type = mapping.get("chart_type")
    output_type = mapping.get("output_type")

    if output_type == "chart":
        supported_chart_types = (
            requirements_context
            .get("requirements", {})
            .get("chart", {})
            .get("supported_chart_types", [])
        )

        if chart_type not in supported_chart_types:
            errors.append(f"Unsupported chart_type: {chart_type}")

    unsupported_terms = mapping.get("unsupported_terms", [])

    if unsupported_terms:
        errors.append(
            "Unsupported terms detected: "
            + ", ".join(str(term) for term in unsupported_terms)
        )

    ambiguous_terms = mapping.get("ambiguous_terms", [])

    if ambiguous_terms:
        errors.append(
            "Ambiguous terms detected: "
            + ", ".join(str(term) for term in ambiguous_terms)
        )

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


def _validate_column(
    column: str | None,
    available_columns: set[str],
    field_name: str,
    errors: list[str],
) -> None:
    if not column:
        return

    if column not in available_columns:
        errors.append(f"Unknown {field_name}: {column}")


def _find_unsupported_filter_values(
    value: Any,
    semantic_context: dict[str, Any],
    is_supported_fn: Any,
) -> list[Any]:
    """
    Validate scalar and IN-list filter values.

    Examples:
        value = "Marconi"
        value = ["Salza", "Marconi"]

    The caller receives only the unsupported values.
    """
    unsupported_values: list[Any] = []

    for item in _iter_filter_values(value):
        if not is_supported_fn(item, semantic_context):
            unsupported_values.append(item)

    return unsupported_values


def _iter_filter_values(value: Any) -> list[Any]:
    """Return a normalized list of filter values."""
    if value is None:
        return []

    if isinstance(value, str):
        stripped_value = value.strip()
        return [stripped_value] if stripped_value else []

    if isinstance(value, (list, tuple, set)):
        normalized_values: list[Any] = []

        for item in value:
            if item is None:
                continue

            if isinstance(item, str):
                stripped_item = item.strip()

                if not stripped_item:
                    continue

                normalized_values.append(stripped_item)
                continue

            normalized_values.append(item)

        return normalized_values

    return [value]


def _is_supported_district(value: Any, semantic_context: dict[str, Any]) -> bool:
    districts = semantic_context.get("locations", {}).get("districts", {})

    return str(value) in districts


def _is_supported_city(value: Any, semantic_context: dict[str, Any]) -> bool:
    cities = semantic_context.get("locations", {}).get("cities", {})

    return str(value) in cities


def _is_supported_measurement_type(value: Any, semantic_context: dict[str, Any]) -> bool:
    measurement_types = semantic_context.get("measurement_types", {})

    for type_data in measurement_types.values():
        if str(value) == str(type_data.get("canonical_value")):
            return True

    return False


def _validate_raw_metric_against_public_schema(
    raw_metric: str | None,
    metric_column: str | None,
    public_schema_context: dict[str, Any],
    errors: list[str],
) -> None:
    if not raw_metric:
        return

    queryable_metrics = public_schema_context.get("queryable_metrics", {})

    matched_metric = _find_queryable_metric(
        raw_metric=raw_metric,
        queryable_metrics=queryable_metrics,
    )

    if matched_metric is None:
        errors.append(f"Unsupported metric requested: {raw_metric}")
        return

    expected_internal_column = matched_metric.get("internal_column")

    if expected_internal_column and metric_column != expected_internal_column:
        errors.append(
            "Metric mapping mismatch: "
            f"raw metric '{raw_metric}' should map to '{expected_internal_column}', "
            f"but got '{metric_column}'."
        )


def _find_queryable_metric(
    raw_metric: str,
    queryable_metrics: dict[str, Any] | list[dict[str, Any]],
) -> dict[str, Any] | None:
    normalized_raw_metric = raw_metric.lower().strip()

    normalized_queryable_metrics = _normalize_queryable_metrics(queryable_metrics)

    for metric_key, metric_data in normalized_queryable_metrics.items():
        candidate_terms = _collect_queryable_metric_terms(
            metric_key=metric_key,
            metric_data=metric_data,
        )

        if normalized_raw_metric in candidate_terms:
            return metric_data

    return None


def _collect_queryable_metric_terms(
    metric_key: str,
    metric_data: dict[str, Any],
) -> set[str]:
    terms = {str(metric_key).lower()}

    label = metric_data.get("label")
    if label:
        terms.add(str(label).lower())

    synonyms = metric_data.get("synonyms", [])

    for synonym in synonyms:
        terms.add(str(synonym).lower())

    return terms


def _normalize_queryable_metrics(
    queryable_metrics: dict[str, Any] | list[dict[str, Any]],
) -> dict[str, Any]:
    if isinstance(queryable_metrics, dict):
        return queryable_metrics

    if not isinstance(queryable_metrics, list):
        return {}

    normalized_metrics: dict[str, Any] = {}

    for metric_item in queryable_metrics:
        if not isinstance(metric_item, dict):
            continue

        metric_key = (
            metric_item.get("concept")
            or metric_item.get("key")
            or metric_item.get("name")
            or metric_item.get("label")
        )

        if not metric_key:
            continue

        normalized_metrics[str(metric_key)] = metric_item

    return normalized_metrics