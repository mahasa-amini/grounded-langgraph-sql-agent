from typing import Any
from functools import lru_cache
from pathlib import Path
import re
import yaml

from agent.state import QueryFilter
from tools.filter_normalization_tool import filter_value_key
from tools.ranking_cleanup_tool import normalize_ranking_text
from agent.state import QueryFilter


def _text_contains_any_synonym(text: str, synonyms: list[str]) -> bool:
    normalized_text = text.lower()

    for synonym in synonyms:
        if synonym.lower() in normalized_text:
            return True

    return False


def _find_business_term_mapping(
    text: str,
    semantic_context: dict[str, Any],
) -> dict[str, Any]:
    business_terms = semantic_context.get("business_terms", {})

    for term_data in business_terms.values():
        synonyms = term_data.get("synonyms", [])

        if _text_contains_any_synonym(text, synonyms):
            return {
                "table": term_data.get("table"),
                "metric_column": term_data.get("metric_column"),
                "time_column": term_data.get("time_column"),
            }

    return {}


def _find_chart_type(
    text: str,
    semantic_context: dict[str, Any],
) -> str | None:
    chart_terms = semantic_context.get("chart_terms", {})

    for chart_data in chart_terms.values():
        synonyms = chart_data.get("synonyms", [])

        if _text_contains_any_synonym(text, synonyms):
            return chart_data.get("chart_type")

    return None


def _find_measurement_type_filter(
    text: str,
    semantic_context: dict[str, Any],
) -> QueryFilter | None:
    measurement_types = semantic_context.get("measurement_types", {})

    for type_data in measurement_types.values():
        canonical_value = type_data.get("canonical_value")
        synonyms = type_data.get("synonyms", [])

        candidates = [canonical_value, *synonyms]

        for candidate in candidates:
            if not candidate:
                continue

            if str(candidate).lower() in text.lower():
                return {
                    "column": "type",
                    "operator": "=",
                    "value": canonical_value,
                }

    return None


def semantic_mapping_tool(
    text: str | None,
    semantic_context: dict[str, Any],
    requirements_context: dict[str, Any],
) -> dict[str, Any]:
    if not text:
        return {}

    mapping_result = _find_business_term_mapping(
        text=text,
        semantic_context=semantic_context,
    )

    chart_type = _find_chart_type(
        text=text,
        semantic_context=semantic_context,
    )

    measurement_type_filter = _find_measurement_type_filter(
        text=text,
        semantic_context=semantic_context,
    )

    defaults = requirements_context.get("defaults", {})

    table_name = mapping_result.get("table") or defaults.get("table")
    metric_column = mapping_result.get("metric_column") or defaults.get("metric_column")
    time_column = mapping_result.get("time_column") or defaults.get("time_column")

    selected_columns = []

    if time_column:
        selected_columns.append(time_column)

    if metric_column:
        selected_columns.append(metric_column)

    filters: list[QueryFilter] = []

    if measurement_type_filter:
        filters.append(measurement_type_filter)

    return {
        "table_name": table_name,
        "metric_column": metric_column,
        "time_column": time_column,
        "selected_columns": selected_columns,
        "chart_type": chart_type,
        "filters": filters,
    }

def find_project_root() -> Path:
    """Find the project root by locating the schema directory."""
    current_path = Path(__file__).resolve()

    for parent in [current_path.parent, *current_path.parents]:
        schema_file_path = parent / "schema" / "semantic_mapping.yaml"

        if schema_file_path.exists():
            return parent

    return Path.cwd()


@lru_cache(maxsize=1)
def load_semantic_mapping_config() -> dict[str, Any]:
    """Load semantic mapping configuration from schema/semantic_mapping.yaml."""
    semantic_mapping_path = (
        find_project_root() / "schema" / "semantic_mapping.yaml"
    )

    if not semantic_mapping_path.exists():
        return {}

    with semantic_mapping_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def semantic_raw_token_exists(
    term: str,
    request_text: str,
) -> bool:
    """Return True when a raw token appears exactly in request text."""
    pattern = rf"(?<![a-zA-Z0-9_]){re.escape(term.lower())}(?![a-zA-Z0-9_])"

    return re.search(
        pattern=pattern,
        string=str(request_text).lower(),
    ) is not None


def semantic_alias_exists_in_request_text(
    alias: Any,
    request_text: str,
) -> bool:
    """
    Match semantic aliases safely.

    Important:
    - snake_case aliases like flow_in must match raw flow_in only
    - flow_in must NOT match normal text like "flow in Marconi"
    """
    if alias is None:
        return False

    alias_text = str(alias).strip()

    if not alias_text:
        return False

    if "_" in alias_text:
        return semantic_raw_token_exists(
            term=alias_text,
            request_text=request_text,
        )

    normalized_alias = normalize_ranking_text(alias_text)
    normalized_request_text = normalize_ranking_text(request_text)

    if not normalized_alias:
        return False

    return f" {normalized_alias} " in f" {normalized_request_text} "


def get_semantic_aliases(
    key: str,
    config: dict[str, Any],
) -> list[str]:
    """Collect aliases for a semantic config item."""
    aliases: list[str] = [key]

    for synonym in config.get("synonyms", []) or []:
        if not synonym:
            continue

        aliases.append(str(synonym))

    return aliases


def get_matching_semantic_aliases(
    key: str,
    config: dict[str, Any],
    request_text: str,
) -> list[str]:
    """Return aliases that actually appear in the request text."""
    matching_aliases: list[str] = []

    for alias in get_semantic_aliases(key=key, config=config):
        if semantic_alias_exists_in_request_text(
            alias=alias,
            request_text=request_text,
        ):
            matching_aliases.append(alias)

    return matching_aliases


def semantic_match_score(alias: str) -> tuple[int, int]:
    """Score aliases by specificity: more tokens and longer text win."""
    normalized_alias = normalize_ranking_text(alias)
    token_count = len(normalized_alias.split())

    return token_count, len(str(alias))


def find_best_business_term_match(
    request_text: str,
    semantic_mapping_config: dict[str, Any],
) -> dict[str, Any] | None:
    """Find the best matching metric business term from semantic_mapping.yaml."""
    business_terms = semantic_mapping_config.get("business_terms", {}) or {}
    matches: list[dict[str, Any]] = []

    for term_key, term_config in business_terms.items():
        if not isinstance(term_config, dict):
            continue

        if term_config.get("kind") != "metric":
            continue

        matching_aliases = get_matching_semantic_aliases(
            key=str(term_key),
            config=term_config,
            request_text=request_text,
        )

        if not matching_aliases:
            continue

        best_alias = sorted(
            matching_aliases,
            key=semantic_match_score,
            reverse=True,
        )[0]

        matches.append(
            {
                "term_key": str(term_key),
                "term_config": term_config,
                "matched_alias": best_alias,
                "matching_aliases": matching_aliases,
                "score": semantic_match_score(best_alias),
            }
        )

    if not matches:
        return None

    return sorted(
        matches,
        key=lambda item: item["score"],
        reverse=True,
    )[0]


def semantic_filter_key(filter_item: dict[str, Any]) -> tuple[str, str, str]:
    """Build a stable key for filter deduplication."""
    column = str(filter_item.get("column") or "").strip().lower()
    operator = str(filter_item.get("operator") or "=").strip().upper()
    value = filter_item.get("value")

    if isinstance(value, (list, tuple, set)):
        normalized_value = ",".join(
            sorted(filter_value_key(item) for item in value)
        )
    else:
        normalized_value = filter_value_key(value)

    return column, operator, normalized_value


def merge_unique_semantic_filters(
    existing_filters: list[dict[str, Any]],
    filters_to_add: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge semantic filters without duplicating existing filters."""
    merged_filters: list[dict[str, Any]] = []
    seen_filter_keys: set[tuple[str, str, str]] = set()

    for filter_item in [*(existing_filters or []), *(filters_to_add or [])]:
        column = filter_item.get("column")

        if not column:
            continue

        normalized_filter = {
            "column": column,
            "operator": str(filter_item.get("operator") or "=").strip().upper(),
            "value": filter_item.get("value"),
        }

        filter_key = semantic_filter_key(normalized_filter)

        if filter_key in seen_filter_keys:
            continue

        seen_filter_keys.add(filter_key)
        merged_filters.append(normalized_filter)

    return merged_filters


def term_cleanup_normalize(value: Any) -> str:
    """Normalize a term for safe request-text comparison."""
    return normalize_ranking_text(str(value))


def add_normalized_resolved_term(
    resolved_terms: set[str],
    term: Any,
) -> None:
    """Add a normalized resolved term to the resolved terms set."""
    normalized_term = term_cleanup_normalize(term)

    if normalized_term:
        resolved_terms.add(normalized_term)


def apply_semantic_mapping_resolution(
    mapping: dict[str, Any],
    request_text: str,
    semantic_mapping_config: dict[str, Any],
    available_columns: set[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Apply deterministic semantic mapping from semantic_mapping.yaml.

    This resolves business terms such as:
        incoming flow -> metric_column=value + filter type=flow_in
    """
    updated_mapping = dict(mapping)
    filters_to_add: list[dict[str, Any]] = []
    resolved_terms: set[str] = set()

    debug_info: dict[str, Any] = {
        "matched_business_term": None,
        "matched_measurement_types": [],
        "added_filters": [],
        "resolved_terms": [],
    }

    business_term_match = find_best_business_term_match(
        request_text=request_text,
        semantic_mapping_config=semantic_mapping_config,
    )

    if business_term_match is not None:
        term_key = business_term_match["term_key"]
        term_config = business_term_match["term_config"]

        debug_info["matched_business_term"] = {
            "term_key": term_key,
            "matched_alias": business_term_match["matched_alias"],
            "matching_aliases": business_term_match["matching_aliases"],
        }

        table_name = term_config.get("table")
        metric_column = term_config.get("metric_column")
        time_column = term_config.get("time_column")

        if table_name:
            updated_mapping["table_name"] = table_name

        if metric_column and (
            available_columns is None or metric_column in available_columns
        ):
            updated_mapping["metric_column"] = metric_column

        if time_column and (
            available_columns is None or time_column in available_columns
        ):
            updated_mapping["time_column"] = time_column

        for alias in business_term_match["matching_aliases"]:
            add_normalized_resolved_term(resolved_terms, alias)

        for semantic_filter in term_config.get("filters", []) or []:
            column = semantic_filter.get("column")

            if not column:
                continue

            if available_columns is not None and column not in available_columns:
                continue

            filter_to_add = {
                "column": column,
                "operator": semantic_filter.get("operator", "="),
                "value": semantic_filter.get("value"),
            }

            filters_to_add.append(filter_to_add)

            add_normalized_resolved_term(
                resolved_terms,
                semantic_filter.get("value"),
            )

    measurement_types = semantic_mapping_config.get("measurement_types", {}) or {}

    for measurement_type_key, measurement_type_config in measurement_types.items():
        if not isinstance(measurement_type_config, dict):
            continue

        matching_aliases = get_matching_semantic_aliases(
            key=str(measurement_type_key),
            config=measurement_type_config,
            request_text=request_text,
        )

        if not matching_aliases:
            continue

        column = measurement_type_config.get("column", "type")
        canonical_value = measurement_type_config.get(
            "canonical_value",
            measurement_type_key,
        )

        if available_columns is not None and column not in available_columns:
            continue

        filter_to_add = {
            "column": column,
            "operator": "=",
            "value": canonical_value,
        }

        filters_to_add.append(filter_to_add)

        debug_info["matched_measurement_types"].append(
            {
                "measurement_type_key": measurement_type_key,
                "canonical_value": canonical_value,
                "matching_aliases": matching_aliases,
            }
        )

        for alias in matching_aliases:
            add_normalized_resolved_term(resolved_terms, alias)

        add_normalized_resolved_term(resolved_terms, canonical_value)

    updated_mapping["filters"] = merge_unique_semantic_filters(
        existing_filters=updated_mapping.get("filters", []) or [],
        filters_to_add=filters_to_add,
    )

    debug_info["added_filters"] = filters_to_add
    debug_info["resolved_terms"] = sorted(resolved_terms)

    return updated_mapping, debug_info