from typing import Any
import re

from tools.schema_dimension_tool import normalize_schema_phrase


def aggregation_alias(aggregation: str | None) -> str | None:
    """Return the SQL alias used for an aggregation."""
    if not aggregation:
        return None

    normalized_aggregation = str(aggregation).strip().lower()

    aliases = {
        "avg": "avg_value",
        "average": "avg_value",
        "mean": "avg_value",
        "sum": "sum_value",
        "total": "sum_value",
        "max": "max_value",
        "maximum": "max_value",
        "min": "min_value",
        "minimum": "min_value",
        "count": "count_value",
    }

    return aliases.get(normalized_aggregation)


def normalize_ranking_text(text: str) -> str:
    """
    Normalize text for ranking keyword detection.
    """
    return " ".join(
        re.sub(r"[^a-zA-Z0-9]+", " ", str(text).lower())
        .strip()
        .split()
    )


def extract_positive_int_after_keyword(
    text: str,
    keyword: str,
) -> int | None:
    """Extract a positive integer after a ranking keyword like top or bottom."""
    normalized_text = normalize_ranking_text(text)
    normalized_keyword = re.escape(normalize_ranking_text(keyword))

    patterns = [
        rf"\b{normalized_keyword}\s+(\d+)\b",
        rf"\b{normalized_keyword}\s*:\s*(\d+)\b",
        rf"\b{normalized_keyword}\s*-\s*(\d+)\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, normalized_text, flags=re.IGNORECASE)

        if match is None:
            continue

        value = int(match.group(1))

        if value > 0:
            return value

    return None


def has_any_term(text: str, terms: list[str]) -> bool:
    """Return True if any whole-word term exists in normalized text."""
    padded_text = f" {text} "

    for term in terms:
        normalized_term = normalize_schema_phrase(term)

        if f" {normalized_term} " in padded_text:
            return True

    return False


def apply_ranking_cleanup_to_mapping(
    mapping: dict[str, Any],
    request_text: str,
) -> dict[str, Any]:
    """
    Apply deterministic ranking behavior for analytical ranking requests.
    """
    cleaned_mapping = dict(mapping)

    aggregation = cleaned_mapping.get("aggregation")
    group_by = cleaned_mapping.get("group_by", []) or []

    if not aggregation or not group_by:
        return cleaned_mapping

    aggregation_order_alias = aggregation_alias(aggregation)

    if aggregation_order_alias is None:
        return cleaned_mapping

    normalized_text = normalize_ranking_text(request_text)

    top_limit = extract_positive_int_after_keyword(
        text=normalized_text,
        keyword="top",
    )
    bottom_limit = extract_positive_int_after_keyword(
        text=normalized_text,
        keyword="bottom",
    )

    descending_ranking_terms = [
        "highest",
        "largest",
        "greatest",
        "most",
    ]

    ascending_ranking_terms = [
        "lowest",
        "smallest",
        "least",
    ]

    max_ranking_terms = [
        "maximum",
        "max",
    ]

    min_ranking_terms = [
        "minimum",
        "min",
    ]

    ordering_terms = [
        "rank",
        "ranking",
        "ordered by",
        "order by",
        "sort by",
        "sorted by",
    ]

    has_top = (
        top_limit is not None
        or re.search(r"\btop\b", normalized_text, flags=re.IGNORECASE) is not None
    )

    has_bottom = (
        bottom_limit is not None
        or re.search(r"\bbottom\b", normalized_text, flags=re.IGNORECASE) is not None
    )

    has_descending_ranking = has_any_term(
        normalized_text,
        descending_ranking_terms,
    )

    has_ascending_ranking = has_any_term(
        normalized_text,
        ascending_ranking_terms,
    )

    has_max_ranking_word = has_any_term(
        normalized_text,
        max_ranking_terms,
    )

    has_min_ranking_word = has_any_term(
        normalized_text,
        min_ranking_terms,
    )

    has_which_intent = " which " in f" {normalized_text} "
    has_with_intent = " with " in f" {normalized_text} "
    has_has_intent = (
        " has " in f" {normalized_text} "
        or " had " in f" {normalized_text} "
    )

    max_min_should_rank = has_which_intent or has_with_intent or has_has_intent

    has_ordering_intent = has_any_term(
        normalized_text,
        ordering_terms,
    )

    if has_top:
        cleaned_mapping["order_by"] = f"{aggregation_order_alias} DESC"
        cleaned_mapping["limit"] = top_limit or 1
        return cleaned_mapping

    if has_bottom:
        cleaned_mapping["order_by"] = f"{aggregation_order_alias} ASC"
        cleaned_mapping["limit"] = bottom_limit or 1
        return cleaned_mapping

    if has_descending_ranking:
        cleaned_mapping["order_by"] = f"{aggregation_order_alias} DESC"
        cleaned_mapping["limit"] = 1
        return cleaned_mapping

    if has_ascending_ranking:
        cleaned_mapping["order_by"] = f"{aggregation_order_alias} ASC"
        cleaned_mapping["limit"] = 1
        return cleaned_mapping

    if has_max_ranking_word and max_min_should_rank:
        cleaned_mapping["order_by"] = f"{aggregation_order_alias} DESC"
        cleaned_mapping["limit"] = 1
        return cleaned_mapping

    if has_min_ranking_word and max_min_should_rank:
        cleaned_mapping["order_by"] = f"{aggregation_order_alias} ASC"
        cleaned_mapping["limit"] = 1
        return cleaned_mapping

    if has_ordering_intent:
        existing_order_by = cleaned_mapping.get("order_by")

        if not existing_order_by:
            cleaned_mapping["order_by"] = f"{aggregation_order_alias} DESC"

        return cleaned_mapping

    return cleaned_mapping