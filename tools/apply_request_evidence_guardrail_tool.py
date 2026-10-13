from difflib import SequenceMatcher
from typing import Any


def apply_request_evidence_guardrail_tool(
    mapping: dict[str, Any],
    request_evidence: dict[str, Any] | None,
    public_schema_context: dict[str, Any],
    semantic_context: dict[str, Any],
    requirements_context: dict[str, Any],
) -> dict[str, Any]:
    """
    Apply deterministic guardrails based on request evidence and schema contexts.

    This tool does not extract evidence from text.
    It only enforces that explicit evidence is not silently ignored by the mapper.
    """

    request_evidence = request_evidence or {}

    updated_mapping = dict(mapping)

    unsupported_terms = _deduplicate_strings(
        [
            str(term)
            for term in updated_mapping.get("unsupported_terms", [])
            if str(term).strip()
        ]
    )
    ambiguous_terms = _deduplicate_strings(
        [
            str(term)
            for term in updated_mapping.get("ambiguous_terms", [])
            if str(term).strip()
        ]
    )
    evidence_suggestions: dict[str, list[str]] = {}

    entity_resolution_config = requirements_context.get("entity_resolution", {})
    fuzzy_config = entity_resolution_config.get("fuzzy_match", {})
    ambiguity_config = entity_resolution_config.get("ambiguity", {})
    location_config = entity_resolution_config.get("locations", {})

    fuzzy_enabled = bool(fuzzy_config.get("enabled", True))
    fuzzy_min_similarity = float(fuzzy_config.get("min_similarity", 0.88))
    fuzzy_auto_apply = bool(fuzzy_config.get("auto_apply", False))
    max_suggestions = int(fuzzy_config.get("max_suggestions", 3))
    treat_fuzzy_as_ambiguous = bool(
        ambiguity_config.get("treat_fuzzy_matches_as_ambiguous", True)
    )

    metric_phrase = _clean_optional_string(request_evidence.get("metric_phrase"))

    if metric_phrase:
        metric_result = _resolve_metric(
            phrase=metric_phrase,
            public_schema_context=public_schema_context,
            fuzzy_enabled=fuzzy_enabled,
            fuzzy_min_similarity=fuzzy_min_similarity,
            max_suggestions=max_suggestions,
        )

        updated_mapping["raw_metric"] = metric_phrase

        if metric_result["status"] == "matched":
            updated_mapping["metric_column"] = metric_result["internal_column"]

            unsupported_terms = _remove_matching_term(
                values=unsupported_terms,
                target=metric_phrase,
            )
            ambiguous_terms = _remove_matching_term(
                values=ambiguous_terms,
                target=metric_phrase,
            )

        elif metric_result["status"] == "fuzzy":
            if fuzzy_auto_apply:
                updated_mapping["metric_column"] = metric_result["internal_column"]
                updated_mapping["raw_metric"] = metric_result["canonical_label"]
            else:
                updated_mapping["metric_column"] = None
                updated_mapping["selected_columns"] = []

                if treat_fuzzy_as_ambiguous:
                    ambiguous_terms = _append_unique(
                        values=ambiguous_terms,
                        value=metric_phrase,
                    )
                    evidence_suggestions[metric_phrase] = metric_result["suggestions"]
                else:
                    unsupported_terms = _append_unique(
                        values=unsupported_terms,
                        value=metric_phrase,
                    )

        else:
            updated_mapping["metric_column"] = None
            updated_mapping["selected_columns"] = []
            unsupported_terms = _append_unique(
                values=unsupported_terms,
                value=metric_phrase,
            )

    city_column = str(location_config.get("city_column", "city"))
    district_column = str(location_config.get("district_column", "district"))
    type_column = "type"

    time_column = _clean_optional_string(updated_mapping.get("time_column")) or "timestamp"
    metric_column = _clean_optional_string(updated_mapping.get("metric_column"))

    untrusted_mapper_filter_columns = _get_untrusted_mapper_filter_columns(
        public_schema_context=public_schema_context,
        fallback_columns=[
            city_column,
            district_column,
            type_column,
            time_column,
            metric_column,
            "sensor_id",
        ],
    )

    location_phrases = [
        phrase.strip()
        for phrase in request_evidence.get("location_phrases", [])
        if isinstance(phrase, str) and phrase.strip()
    ]

    filters = _sanitize_filters(
        filters=updated_mapping.get("filters", []),
        semantic_context=semantic_context,
        city_column=city_column,
        district_column=district_column,
        type_column=type_column,
    )

    filters = _remove_ungrounded_mapper_filters(
        filters=filters,
        columns_to_remove=untrusted_mapper_filter_columns,
    )

    location_resolution = _resolve_locations(
        location_phrases=location_phrases,
        semantic_context=semantic_context,
        requirements_context=requirements_context,
        fuzzy_enabled=fuzzy_enabled,
        fuzzy_min_similarity=fuzzy_min_similarity,
        fuzzy_auto_apply=fuzzy_auto_apply,
        treat_fuzzy_as_ambiguous=treat_fuzzy_as_ambiguous,
        max_suggestions=max_suggestions,
        city_column=city_column,
        district_column=district_column,
    )

    filters.extend(location_resolution["filters"])
    filters = _deduplicate_filters(filters)

    unsupported_terms = _deduplicate_strings(
        [
            *unsupported_terms,
            *location_resolution["unsupported_terms"],
        ]
    )

    ambiguous_terms = _deduplicate_strings(
        [
            *ambiguous_terms,
            *location_resolution["ambiguous_terms"],
        ]
    )

    for phrase, suggestions in location_resolution["suggestions"].items():
        evidence_suggestions[phrase] = suggestions

    updated_mapping["filters"] = filters
    updated_mapping["unsupported_terms"] = unsupported_terms
    updated_mapping["ambiguous_terms"] = ambiguous_terms

    metric_column = updated_mapping.get("metric_column")
    time_column = updated_mapping.get("time_column")

    if unsupported_terms or ambiguous_terms:
        if metric_phrase and updated_mapping.get("metric_column") is None:
            updated_mapping["selected_columns"] = []
    elif not updated_mapping.get("selected_columns") and metric_column and time_column:
        updated_mapping["selected_columns"] = [time_column, metric_column]

    return {
        "mapping": updated_mapping,
        "evidence_suggestions": evidence_suggestions,
    }


def _remove_location_filters(
    filters: list[dict[str, Any]],
    city_column: str,
    district_column: str,
) -> list[dict[str, Any]]:
    return [
        filter_item
        for filter_item in filters
        if filter_item.get("column") not in {city_column, district_column}
    ]

def _remove_ungrounded_mapper_filters(
    filters: list[dict[str, Any]],
    columns_to_remove: list[Any],
) -> list[dict[str, Any]]:
    removable_columns = {
        str(column).strip()
        for column in columns_to_remove
        if isinstance(column, str) and column.strip()
    }

    return [
        filter_item
        for filter_item in filters
        if filter_item.get("column") not in removable_columns
    ]


def _get_untrusted_mapper_filter_columns(
    public_schema_context: dict[str, Any],
    fallback_columns: list[Any],
) -> list[str]:
    allowed_filter_columns = public_schema_context.get("allowed_filter_columns", [])

    schema_filter_columns: list[str] = []

    if isinstance(allowed_filter_columns, list):
        schema_filter_columns = [
            str(column).strip()
            for column in allowed_filter_columns
            if str(column).strip()
        ]

    return _deduplicate_strings(
        [
            *schema_filter_columns,
            *[
                str(column).strip()
                for column in fallback_columns
                if isinstance(column, str) and column.strip()
            ],
        ]
    )

def _resolve_metric(
    phrase: str,
    public_schema_context: dict[str, Any],
    fuzzy_enabled: bool,
    fuzzy_min_similarity: float,
    max_suggestions: int,
) -> dict[str, Any]:
    metric_registry = _build_metric_registry(public_schema_context)
    normalized_phrase = _normalize_text(phrase)

    for metric_item in metric_registry:
        if normalized_phrase in metric_item["normalized_terms"]:
            return {
                "status": "matched",
                "canonical_label": metric_item["canonical_label"],
                "internal_column": metric_item["internal_column"],
                "suggestions": [],
            }

    if fuzzy_enabled:
        fuzzy_matches = _find_fuzzy_matches(
            phrase=phrase,
            candidates=metric_registry,
            fuzzy_min_similarity=fuzzy_min_similarity,
            max_suggestions=max_suggestions,
        )

        if fuzzy_matches:
            best_match = fuzzy_matches[0]
            return {
                "status": "fuzzy",
                "canonical_label": best_match["canonical_label"],
                "internal_column": best_match["internal_column"],
                "suggestions": [
                    match["canonical_label"]
                    for match in fuzzy_matches
                ],
            }

    return {
        "status": "unsupported",
        "canonical_label": None,
        "internal_column": None,
        "suggestions": [],
    }


def _resolve_locations(
    location_phrases: list[str],
    semantic_context: dict[str, Any],
    requirements_context: dict[str, Any],
    fuzzy_enabled: bool,
    fuzzy_min_similarity: float,
    fuzzy_auto_apply: bool,
    treat_fuzzy_as_ambiguous: bool,
    max_suggestions: int,
    city_column: str,
    district_column: str,
) -> dict[str, Any]:
    location_registry = _build_location_registry(
        semantic_context=semantic_context,
        city_column=city_column,
        district_column=district_column,
    )

    location_config = (
        requirements_context
        .get("entity_resolution", {})
        .get("locations", {})
    )

    reject_city_district_mismatch = bool(
        location_config.get("reject_city_district_mismatch", True)
    )
    add_parent_city_filter_for_district = bool(
        location_config.get("add_parent_city_filter_for_district", False)
    )

    filters: list[dict[str, Any]] = []
    unsupported_terms: list[str] = []
    ambiguous_terms: list[str] = []
    suggestions: dict[str, list[str]] = {}

    matched_city_values: set[str] = set()
    matched_district_items: list[dict[str, Any]] = []

    exact_matched_values = _collect_exact_matched_location_values(
        location_phrases=location_phrases,
        location_registry=location_registry,
    )

    for phrase in location_phrases:
        normalized_phrase = _normalize_text(phrase)

        exact_matches = [
            item
            for item in location_registry
            if normalized_phrase in item["normalized_terms"]
        ]

        if exact_matches:
            exact_match = exact_matches[0]

            if exact_match["entity_type"] == "city":
                matched_city_values.add(exact_match["canonical_value"])
                filters.append(
                    {
                        "column": city_column,
                        "operator": "=",
                        "value": exact_match["canonical_value"],
                    }
                )

            elif exact_match["entity_type"] == "district":
                matched_district_items.append(exact_match)
                filters.append(
                    {
                        "column": district_column,
                        "operator": "=",
                        "value": exact_match["canonical_value"],
                    }
                )

                parent_city = exact_match.get("parent_city")
                if add_parent_city_filter_for_district and parent_city:
                    filters.append(
                        {
                            "column": city_column,
                            "operator": "=",
                            "value": parent_city,
                        }
                    )

            continue

        fuzzy_matches = []

        if fuzzy_enabled:
            fuzzy_matches = _find_fuzzy_matches(
                phrase=phrase,
                candidates=location_registry,
                fuzzy_min_similarity=fuzzy_min_similarity,
                max_suggestions=max_suggestions,
            )

        if fuzzy_matches:
            fuzzy_suggestion_values = {
                _normalize_text(str(match["canonical_value"]))
                for match in fuzzy_matches
            }

            if fuzzy_suggestion_values.intersection(exact_matched_values):
                continue

            if fuzzy_auto_apply:
                best_match = fuzzy_matches[0]

                if best_match["entity_type"] == "city":
                    matched_city_values.add(best_match["canonical_value"])
                    filters.append(
                        {
                            "column": city_column,
                            "operator": "=",
                            "value": best_match["canonical_value"],
                        }
                    )

                elif best_match["entity_type"] == "district":
                    matched_district_items.append(best_match)
                    filters.append(
                        {
                            "column": district_column,
                            "operator": "=",
                            "value": best_match["canonical_value"],
                        }
                    )

            elif treat_fuzzy_as_ambiguous:
                ambiguous_terms = _append_unique(
                    values=ambiguous_terms,
                    value=phrase,
                )
                suggestions[phrase] = [
                    match["canonical_value"]
                    for match in fuzzy_matches
                ]
            else:
                unsupported_terms = _append_unique(
                    values=unsupported_terms,
                    value=phrase,
                )

            continue

        unsupported_terms = _append_unique(
            values=unsupported_terms,
            value=phrase,
        )

    if reject_city_district_mismatch and matched_city_values and matched_district_items:
        for district_item in matched_district_items:
            parent_city = district_item.get("parent_city")

            if parent_city and parent_city not in matched_city_values:
                mismatch_term = (
                    f"{parent_city}:{district_item['canonical_value']}"
                )
                unsupported_terms = _append_unique(
                    values=unsupported_terms,
                    value=mismatch_term,
                )

    return {
        "filters": _deduplicate_filters(filters),
        "unsupported_terms": unsupported_terms,
        "ambiguous_terms": ambiguous_terms,
        "suggestions": suggestions,
    }


def _collect_exact_matched_location_values(
    location_phrases: list[str],
    location_registry: list[dict[str, Any]],
) -> set[str]:
    exact_matched_values: set[str] = set()

    for phrase in location_phrases:
        normalized_phrase = _normalize_text(phrase)

        for item in location_registry:
            if normalized_phrase in item["normalized_terms"]:
                exact_matched_values.add(
                    _normalize_text(str(item["canonical_value"]))
                )

    return exact_matched_values


def _sanitize_filters(
    filters: list[dict[str, Any]],
    semantic_context: dict[str, Any],
    city_column: str,
    district_column: str,
    type_column: str,
) -> list[dict[str, Any]]:
    sanitized_filters: list[dict[str, Any]] = []

    for filter_item in filters:
        column = filter_item.get("column")
        value = filter_item.get("value")

        if column == city_column:
            if _is_supported_city(value, semantic_context):
                sanitized_filters.append(filter_item)
            continue

        if column == district_column:
            if _is_supported_district(value, semantic_context):
                sanitized_filters.append(filter_item)
            continue

        if column == type_column:
            if _is_supported_measurement_type(value, semantic_context):
                sanitized_filters.append(filter_item)
            continue

        sanitized_filters.append(filter_item)

    return sanitized_filters


def _build_metric_registry(
    public_schema_context: dict[str, Any],
) -> list[dict[str, Any]]:
    queryable_metrics = public_schema_context.get("queryable_metrics", [])
    metric_items = _normalize_queryable_metrics(queryable_metrics)

    registry: list[dict[str, Any]] = []

    for metric_key, metric_data in metric_items.items():
        terms = _collect_metric_terms(
            metric_key=metric_key,
            metric_data=metric_data,
        )

        internal_column = metric_data.get("internal_column")

        if not internal_column:
            continue

        canonical_label = (
            metric_data.get("label")
            or metric_data.get("concept")
            or metric_key
        )

        registry.append(
            {
                "entity_type": "metric",
                "canonical_value": str(metric_key),
                "canonical_label": str(canonical_label),
                "internal_column": str(internal_column),
                "terms": terms,
                "normalized_terms": {
                    _normalize_text(term)
                    for term in terms
                    if str(term).strip()
                },
            }
        )

    return registry


def _build_location_registry(
    semantic_context: dict[str, Any],
    city_column: str,
    district_column: str,
) -> list[dict[str, Any]]:
    locations = semantic_context.get("locations", {})

    cities = locations.get("cities", {})
    districts = locations.get("districts", {})

    registry: list[dict[str, Any]] = []

    for city_key, city_data in cities.items():
        canonical_value = str(city_data.get("canonical_value") or city_key)
        terms = _collect_value_terms(
            item_key=str(city_key),
            item_data=city_data,
        )

        registry.append(
            {
                "entity_type": "city",
                "column": city_column,
                "canonical_value": canonical_value,
                "canonical_label": canonical_value,
                "parent_city": None,
                "terms": terms,
                "normalized_terms": {
                    _normalize_text(term)
                    for term in terms
                    if str(term).strip()
                },
            }
        )

    for district_key, district_data in districts.items():
        canonical_value = str(district_data.get("canonical_value") or district_key)
        parent_city = district_data.get("city")

        terms = _collect_value_terms(
            item_key=str(district_key),
            item_data=district_data,
        )

        registry.append(
            {
                "entity_type": "district",
                "column": district_column,
                "canonical_value": canonical_value,
                "canonical_label": canonical_value,
                "parent_city": str(parent_city) if parent_city else None,
                "terms": terms,
                "normalized_terms": {
                    _normalize_text(term)
                    for term in terms
                    if str(term).strip()
                },
            }
        )

    return registry


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


def _collect_metric_terms(
    metric_key: str,
    metric_data: dict[str, Any],
) -> set[str]:
    terms = {str(metric_key)}

    for field_name in ["concept", "key", "name", "label"]:
        field_value = metric_data.get(field_name)
        if field_value:
            terms.add(str(field_value))

    synonyms = metric_data.get("synonyms", [])

    for synonym in synonyms:
        if synonym:
            terms.add(str(synonym))

    return terms


def _collect_value_terms(
    item_key: str,
    item_data: dict[str, Any],
) -> set[str]:
    terms = {item_key}

    canonical_value = item_data.get("canonical_value")
    if canonical_value:
        terms.add(str(canonical_value))

    label = item_data.get("label")
    if label:
        terms.add(str(label))

    synonyms = item_data.get("synonyms", [])
    for synonym in synonyms:
        if synonym:
            terms.add(str(synonym))

    return terms


def _find_fuzzy_matches(
    phrase: str,
    candidates: list[dict[str, Any]],
    fuzzy_min_similarity: float,
    max_suggestions: int,
) -> list[dict[str, Any]]:
    normalized_phrase = _normalize_text(phrase)

    scored_matches: list[tuple[float, dict[str, Any]]] = []

    for candidate in candidates:
        best_score = 0.0

        for term in candidate["normalized_terms"]:
            score = SequenceMatcher(None, normalized_phrase, term).ratio()
            best_score = max(best_score, score)

        if best_score >= fuzzy_min_similarity:
            scored_matches.append((best_score, candidate))

    scored_matches.sort(key=lambda item: item[0], reverse=True)

    unique_matches: list[dict[str, Any]] = []
    seen_labels: set[str] = set()

    for _, candidate in scored_matches:
        label = candidate["canonical_label"]

        if label in seen_labels:
            continue

        seen_labels.add(label)
        unique_matches.append(candidate)

        if len(unique_matches) >= max_suggestions:
            break

    return unique_matches


def _is_supported_city(
    value: Any,
    semantic_context: dict[str, Any],
) -> bool:
    if value is None:
        return False

    cities = semantic_context.get("locations", {}).get("cities", {})
    normalized_value = _normalize_text(str(value))

    for city_key, city_data in cities.items():
        terms = _collect_value_terms(
            item_key=str(city_key),
            item_data=city_data,
        )

        if normalized_value in {_normalize_text(term) for term in terms}:
            return True

    return False


def _is_supported_district(
    value: Any,
    semantic_context: dict[str, Any],
) -> bool:
    if value is None:
        return False

    districts = semantic_context.get("locations", {}).get("districts", {})
    normalized_value = _normalize_text(str(value))

    for district_key, district_data in districts.items():
        terms = _collect_value_terms(
            item_key=str(district_key),
            item_data=district_data,
        )

        if normalized_value in {_normalize_text(term) for term in terms}:
            return True

    return False


def _is_supported_measurement_type(
    value: Any,
    semantic_context: dict[str, Any],
) -> bool:
    if value is None:
        return False

    measurement_types = semantic_context.get("measurement_types", {})
    normalized_value = _normalize_text(str(value))

    for type_key, type_data in measurement_types.items():
        terms = _collect_value_terms(
            item_key=str(type_key),
            item_data=type_data,
        )

        if normalized_value in {_normalize_text(term) for term in terms}:
            return True

    return False


def _deduplicate_filters(
    filters: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    unique_filters: list[dict[str, Any]] = []
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


def _deduplicate_strings(values: list[str]) -> list[str]:
    unique_values: list[str] = []
    seen = set()

    for value in values:
        cleaned_value = str(value).strip()

        if not cleaned_value:
            continue

        normalized_value = _normalize_text(cleaned_value)

        if normalized_value in seen:
            continue

        seen.add(normalized_value)
        unique_values.append(cleaned_value)

    return unique_values


def _append_unique(values: list[str], value: str) -> list[str]:
    return _deduplicate_strings([*values, value])


def _remove_matching_term(values: list[str], target: str) -> list[str]:
    normalized_target = _normalize_text(target)

    return [
        value
        for value in values
        if _normalize_text(value) != normalized_target
    ]


def _clean_optional_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None

    cleaned_value = value.strip()

    if not cleaned_value:
        return None

    return cleaned_value


def _normalize_text(value: str) -> str:
    return " ".join(value.lower().strip().split())