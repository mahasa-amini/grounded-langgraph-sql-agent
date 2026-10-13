from typing import Any

from agent.state import QueryFilter


def _normalize_exact_value(value: Any) -> str:
    return " ".join(str(value or "").replace("_", " ").lower().strip().split())


def _find_match_in_mapping(
    text: str,
    mapping_items: dict[str, Any],
    column_name: str,
) -> QueryFilter | None:
    normalized_text = text.lower()

    for item_data in mapping_items.values():
        canonical_value = item_data.get("canonical_value")
        synonyms = item_data.get("synonyms", [])

        candidates = [canonical_value, *synonyms]

        for candidate in candidates:
            if not candidate:
                continue

            if str(candidate).lower() in normalized_text:
                return {
                    "column": column_name,
                    "operator": "=",
                    "value": canonical_value,
                }

    return None


def normalize_location_tool(
    text: str | None,
    semantic_context: dict[str, Any],
) -> list[QueryFilter]:
    if not text:
        return []

    locations = semantic_context.get("locations", {})

    cities = locations.get("cities", {})
    districts = locations.get("districts", {})

    filters: list[QueryFilter] = []

    city_filter = _find_match_in_mapping(
        text=text,
        mapping_items=cities,
        column_name="city",
    )

    if city_filter:
        filters.append(city_filter)

    district_filter = _find_match_in_mapping(
        text=text,
        mapping_items=districts,
        column_name="district",
    )

    if district_filter:
        filters.append(district_filter)

    return filters


def normalize_exact_location_tool(
    text: str | None,
    semantic_context: dict[str, Any],
) -> list[QueryFilter]:
    """Resolve a whole-message location only when it has one schema-backed match."""
    normalized_text = _normalize_exact_value(text)
    if not normalized_text:
        return []

    matches: list[QueryFilter] = []
    locations = semantic_context.get("locations", {}) or {}
    for collection_name, column_name in (("cities", "city"), ("districts", "district")):
        for key, item_data in (locations.get(collection_name, {}) or {}).items():
            if not isinstance(item_data, dict):
                continue
            canonical_value = item_data.get("canonical_value") or key
            candidates = [key, canonical_value, *(item_data.get("synonyms", []) or [])]
            if normalized_text in {_normalize_exact_value(value) for value in candidates if value}:
                matches.append({
                    "column": column_name,
                    "operator": "=",
                    "value": canonical_value,
                })

    unique = {
        (item["column"], str(item["value"])): item
        for item in matches
    }
    return list(unique.values()) if len(unique) == 1 else []
