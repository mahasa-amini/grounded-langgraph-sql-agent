from typing import Any


def normalize_schema_phrase(value: Any) -> str:
    """Normalize a user/schema phrase for safe comparison."""
    return " ".join(
        str(value)
        .replace("_", " ")
        .replace("-", " ")
        .lower()
        .strip()
        .split()
    )


def pluralize_schema_term(term: str) -> str:
    """Create a simple plural form for schema dimension aliases."""
    if term.endswith("y"):
        return term[:-1] + "ies"

    if term.endswith("s"):
        return term

    return term + "s"


def build_schema_dimension_aliases(dimension_columns: set[str]) -> set[str]:
    """
    Build aliases for schema dimensions that may appear in user text.
    """
    aliases: set[str] = set()

    for column in dimension_columns:
        normalized_column = normalize_schema_phrase(column)

        if not normalized_column:
            continue

        aliases.add(normalized_column)
        aliases.add(pluralize_schema_term(normalized_column))

        raw_column = str(column).lower().strip()
        if raw_column:
            aliases.add(raw_column)
            aliases.add(pluralize_schema_term(raw_column))

        if normalized_column.endswith(" id"):
            without_id = normalized_column[:-3].strip()

            if without_id:
                aliases.add(without_id)
                aliases.add(pluralize_schema_term(without_id))

        if normalized_column == "city":
            aliases.add("cities")

        if normalized_column == "district":
            aliases.add("districts")

        if normalized_column == "type":
            aliases.add("types")
            aliases.add("measurement type")
            aliases.add("measurement types")

    return aliases


def collect_schema_dimension_columns_from_mapping(
    mapping: dict[str, Any],
    available_columns: set[str] | None = None,
) -> set[str]:
    """
    Collect schema columns that are being used as analytical dimensions.
    """
    dimension_columns: set[str] = set()

    for column in mapping.get("selected_columns", []):
        if not column:
            continue

        if available_columns is not None and column not in available_columns:
            continue

        dimension_columns.add(str(column))

    for column in mapping.get("group_by", []):
        if not column:
            continue

        if available_columns is not None and column not in available_columns:
            continue

        dimension_columns.add(str(column))

    return dimension_columns


def remove_schema_dimension_location_phrases(
    request_evidence: dict[str, Any] | None,
    dimension_columns: set[str],
) -> dict[str, Any]:
    """
    Remove schema dimension words from location_phrases.
    """
    if not request_evidence:
        return {}

    cleaned_evidence = dict(request_evidence)

    location_phrases = cleaned_evidence.get("location_phrases", []) or []
    dimension_aliases = build_schema_dimension_aliases(dimension_columns)

    cleaned_location_phrases: list[str] = []

    for phrase in location_phrases:
        normalized_phrase = normalize_schema_phrase(phrase)

        if normalized_phrase in dimension_aliases:
            continue

        cleaned_location_phrases.append(phrase)

    cleaned_evidence["location_phrases"] = cleaned_location_phrases

    return cleaned_evidence


def remove_schema_dimension_unsupported_terms(
    unsupported_terms: list[Any],
    dimension_columns: set[str],
) -> list[Any]:
    """
    Remove unsupported terms that are actually schema dimensions used for grouping.
    """
    dimension_aliases = build_schema_dimension_aliases(dimension_columns)

    cleaned_terms: list[Any] = []

    for term in unsupported_terms:
        normalized_term = normalize_schema_phrase(term)

        if normalized_term in dimension_aliases:
            continue

        cleaned_terms.append(term)

    return cleaned_terms

def detect_requested_group_by_columns(request_text: str) -> list[str]:
    """
    Detects schema dimension columns that the user explicitly asks to group by.

    This is deterministic text-based logic used before schema mapping guardrails.
    It detects phrases such as:
    - by district
    - for each sensor
    - top districts
    - rank by measurement type
    """
    normalized_request_text = normalize_schema_phrase(request_text)

    requested_group_by_columns: list[str] = []

    grouping_patterns_by_column = {
        "district": [
            "by district",
            "by districts",
            "per district",
            "per districts",
            "for each district",
            "for each districts",
            "every district",
            "all districts",
            "across district",
            "across districts",
            "group by district",
            "group by districts",
            "grouped by district",
            "grouped by districts",
            "which district",
            "which districts",
            "top district",
            "top districts",
            "bottom district",
            "bottom districts",
            "rank district",
            "rank districts",
            "ranking district",
            "ranking districts",
        ],
        "city": [
            "by city",
            "by cities",
            "per city",
            "per cities",
            "for each city",
            "for each cities",
            "every city",
            "all cities",
            "across city",
            "across cities",
            "group by city",
            "group by cities",
            "grouped by city",
            "grouped by cities",
            "which city",
            "which cities",
            "top city",
            "top cities",
            "bottom city",
            "bottom cities",
            "rank city",
            "rank cities",
            "ranking city",
            "ranking cities",
        ],
        "sensor_id": [
            "by sensor",
            "by sensors",
            "by sensor id",
            "by sensor ids",
            "per sensor",
            "per sensors",
            "per sensor id",
            "per sensor ids",
            "for each sensor",
            "for each sensor id",
            "every sensor",
            "all sensors",
            "across sensors",
            "group by sensor",
            "group by sensor id",
            "which sensor",
            "which sensor id",
            "top sensor",
            "top sensors",
            "bottom sensor",
            "bottom sensors",
            "rank sensor",
            "rank sensors",
        ],
        "type": [
            "by type",
            "by types",
            "by measurement type",
            "by measurement types",
            "per type",
            "per types",
            "for each type",
            "every type",
            "all types",
            "across types",
            "group by type",
            "group by measurement type",
            "which type",
            "which measurement type",
            "top type",
            "bottom type",
            "rank type",
            "rank types",
        ],
    }

    for column, patterns in grouping_patterns_by_column.items():
        if any(pattern in normalized_request_text for pattern in patterns):
            requested_group_by_columns.append(column)

    ranking_terms = [
        "highest",
        "lowest",
        "maximum",
        "minimum",
        "max",
        "min",
        "top",
        "bottom",
        "rank",
        "ranking",
        "ordered by",
        "sort by",
    ]

    has_ranking_intent = any(
        term in normalized_request_text
        for term in ranking_terms
    )

    if has_ranking_intent:
        if "district" in normalized_request_text and "district" not in requested_group_by_columns:
            requested_group_by_columns.append("district")

        if "city" in normalized_request_text and "city" not in requested_group_by_columns:
            requested_group_by_columns.append("city")

        if (
            "sensor" in normalized_request_text
            and "sensor_id" not in requested_group_by_columns
        ):
            requested_group_by_columns.append("sensor_id")

        if (
            "measurement type" in normalized_request_text
            or " type " in f" {normalized_request_text} "
        ):
            if "type" not in requested_group_by_columns:
                requested_group_by_columns.append("type")

    return requested_group_by_columns

def remove_requested_group_by_phrases_from_locations(
    location_phrases: object,
    requested_group_by_columns: list[str],
) -> list[str]:
    """
    Removes schema dimension phrases from location evidence when those dimensions
    were explicitly requested as group-by columns.

    Example:
    - location_phrases=["district"], requested_group_by_columns=["district"] -> []
    - location_phrases=["Marconi"], requested_group_by_columns=["district"] -> ["Marconi"]
    """
    if not isinstance(location_phrases, list):
        return []

    requested_group_by_aliases = build_schema_dimension_aliases(
        set(requested_group_by_columns)
    )

    cleaned_location_phrases: list[str] = []

    for phrase in location_phrases:
        if not isinstance(phrase, str):
            continue

        normalized_phrase = normalize_schema_phrase(phrase)

        if normalized_phrase in requested_group_by_aliases:
            continue

        cleaned_location_phrases.append(phrase)

    return cleaned_location_phrases