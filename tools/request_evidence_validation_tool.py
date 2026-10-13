from functools import lru_cache
from pathlib import Path
from typing import Any
from difflib import SequenceMatcher
import re

import yaml


def _normalize_semantic_text(value: Any) -> str:
    return " ".join(
        re.sub(r"[^a-zA-Z0-9]+", " ", str(value or "").lower())
        .strip()
        .split()
    )


def _find_project_root() -> Path:
    current_path = Path(__file__).resolve()

    for parent in [current_path.parent, *current_path.parents]:
        schema_file_path = parent / "schema" / "semantic_mapping.yaml"

        if schema_file_path.exists():
            return parent

    return Path.cwd()


@lru_cache(maxsize=1)
def _load_semantic_mapping_config() -> dict[str, Any]:
    semantic_mapping_path = (
        _find_project_root() / "schema" / "semantic_mapping.yaml"
    )

    if not semantic_mapping_path.exists():
        return {}

    with semantic_mapping_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _semantic_raw_token_exists(term: str, request_text: str) -> bool:
    pattern = rf"(?<![a-zA-Z0-9_]){re.escape(term.lower())}(?![a-zA-Z0-9_])"

    return re.search(
        pattern=pattern,
        string=str(request_text).lower(),
    ) is not None


def _semantic_alias_exists_in_request_text(
    alias: Any,
    request_text: str,
) -> bool:
    if alias is None:
        return False

    alias_text = str(alias).strip()

    if not alias_text:
        return False

    if "_" in alias_text:
        return _semantic_raw_token_exists(
            term=alias_text,
            request_text=request_text,
        )

    normalized_alias = _normalize_semantic_text(alias_text)
    normalized_request_text = _normalize_semantic_text(request_text)

    if not normalized_alias:
        return False

    return f" {normalized_alias} " in f" {normalized_request_text} "


def _get_semantic_aliases(
    key: str,
    config: dict[str, Any],
) -> list[str]:
    aliases: list[str] = [key]

    for synonym in config.get("synonyms", []) or []:
        if not synonym:
            continue

        aliases.append(str(synonym))

    return aliases


def _get_matching_semantic_aliases(
    key: str,
    config: dict[str, Any],
    request_text: str,
) -> list[str]:
    matching_aliases: list[str] = []

    for alias in _get_semantic_aliases(key=key, config=config):
        if _semantic_alias_exists_in_request_text(
            alias=alias,
            request_text=request_text,
        ):
            matching_aliases.append(alias)

    return matching_aliases


def _semantic_match_score(alias: str) -> tuple[int, int]:
    normalized_alias = _normalize_semantic_text(alias)
    token_count = len(normalized_alias.split())

    return token_count, len(str(alias))


def _find_best_business_term_match(
    request_text: str,
    semantic_mapping_config: dict[str, Any],
) -> dict[str, Any] | None:
    business_terms = semantic_mapping_config.get("business_terms", {}) or {}
    matches: list[dict[str, Any]] = []

    for term_key, term_config in business_terms.items():
        if not isinstance(term_config, dict):
            continue

        if term_config.get("kind") != "metric":
            continue

        matching_aliases = _get_matching_semantic_aliases(
            key=str(term_key),
            config=term_config,
            request_text=request_text,
        )

        if not matching_aliases:
            continue

        best_alias = sorted(
            matching_aliases,
            key=_semantic_match_score,
            reverse=True,
        )[0]

        matches.append(
            {
                "term_key": str(term_key),
                "term_config": term_config,
                "matched_alias": best_alias,
                "matching_aliases": matching_aliases,
                "score": _semantic_match_score(best_alias),
            }
        )

    if not matches:
        return None

    return sorted(
        matches,
        key=lambda item: item["score"],
        reverse=True,
    )[0]


def _collect_semantic_dimension_aliases(
    semantic_mapping_config: dict[str, Any],
) -> set[str]:
    aliases: set[str] = {
        "city",
        "cities",
        "district",
        "districts",
        "area",
        "areas",
        "zone",
        "zones",
        "neighborhood",
        "neighborhoods",
        "sensor",
        "sensors",
        "sensor id",
        "type",
        "types",
        "measurement type",
        "measurement types",
    }

    business_terms = semantic_mapping_config.get("business_terms", {}) or {}

    for term_key, term_config in business_terms.items():
        if not isinstance(term_config, dict):
            continue

        if term_config.get("kind") != "dimension":
            continue

        aliases.add(_normalize_semantic_text(term_key))

        for synonym in term_config.get("synonyms", []) or []:
            aliases.add(_normalize_semantic_text(synonym))

    return {alias for alias in aliases if alias}


def _iter_semantic_location_entries(
    semantic_mapping_config: dict[str, Any],
) -> list[dict[str, Any]]:
    locations = semantic_mapping_config.get("locations", {}) or {}
    entries: list[dict[str, Any]] = []

    cities = locations.get("cities", {}) or {}

    for city_key, city_config in cities.items():
        if not isinstance(city_config, dict):
            continue

        canonical_value = city_config.get("canonical_value", city_key)
        aliases = [canonical_value, city_key, *(city_config.get("synonyms", []) or [])]

        entries.append(
            {
                "kind": "city",
                "canonical_value": canonical_value,
                "city": canonical_value,
                "aliases": [str(alias) for alias in aliases if alias],
            }
        )

    districts = locations.get("districts", {}) or {}

    for district_key, district_config in districts.items():
        if not isinstance(district_config, dict):
            continue

        canonical_value = district_config.get("canonical_value", district_key)
        aliases = [
            canonical_value,
            district_key,
            *(district_config.get("synonyms", []) or []),
        ]

        entries.append(
            {
                "kind": "district",
                "canonical_value": canonical_value,
                "city": district_config.get("city"),
                "aliases": [str(alias) for alias in aliases if alias],
            }
        )

    return entries


def _resolve_location_phrase_from_semantic_mapping(
    location_phrase: str,
    semantic_mapping_config: dict[str, Any],
) -> dict[str, Any]:
    normalized_phrase = _normalize_semantic_text(location_phrase)

    if not normalized_phrase:
        return {"status": "ignored_dimension"}

    dimension_aliases = _collect_semantic_dimension_aliases(
        semantic_mapping_config=semantic_mapping_config,
    )

    if normalized_phrase in dimension_aliases:
        return {"status": "ignored_dimension"}

    entries = _iter_semantic_location_entries(
        semantic_mapping_config=semantic_mapping_config,
    )

    exact_matches: list[dict[str, Any]] = []

    for entry in entries:
        for alias in entry.get("aliases", []):
            if _normalize_semantic_text(alias) == normalized_phrase:
                exact_matches.append(entry)
                break

    if len(exact_matches) == 1:
        return {
            "status": "resolved",
            "location": exact_matches[0],
        }

    if len(exact_matches) > 1:
        return {
            "status": "ambiguous",
            "matches": exact_matches,
        }

    fuzzy_candidates: list[dict[str, Any]] = []

    for entry in entries:
        for alias in entry.get("aliases", []):
            normalized_alias = _normalize_semantic_text(alias)

            if not normalized_alias:
                continue

            similarity = SequenceMatcher(
                None,
                normalized_phrase,
                normalized_alias,
            ).ratio()

            if similarity >= 0.82:
                fuzzy_candidates.append(
                    {
                        "entry": entry,
                        "alias": alias,
                        "similarity": similarity,
                    }
                )

    if fuzzy_candidates:
        best_candidate = sorted(
            fuzzy_candidates,
            key=lambda item: item["similarity"],
            reverse=True,
        )[0]

        return {
            "status": "fuzzy",
            "suggestion": best_candidate["entry"],
            "matched_alias": best_candidate["alias"],
            "similarity": best_candidate["similarity"],
        }

    return {
        "status": "unsupported",
        "term": location_phrase,
    }


def _is_metric_phrase_supported_by_semantic_mapping(
    metric_phrase: str | None,
    request_text: str,
    semantic_mapping_config: dict[str, Any],
) -> bool:
    if not metric_phrase:
        return True

    business_term_match = _find_best_business_term_match(
        request_text=request_text,
        semantic_mapping_config=semantic_mapping_config,
    )

    return business_term_match is not None


def validate_request_evidence_against_catalog_tool(
    request_evidence: dict[str, Any] | None,
    request_text: str,
) -> dict[str, Any]:
    semantic_mapping_config = _load_semantic_mapping_config()
    request_evidence = request_evidence or {}

    unsupported_terms: list[str] = []
    ambiguous_terms: list[str] = []
    evidence_suggestions: dict[str, Any] = {}

    metric_phrase = request_evidence.get("metric_phrase")

    if metric_phrase and not _is_metric_phrase_supported_by_semantic_mapping(
        metric_phrase=metric_phrase,
        request_text=request_text,
        semantic_mapping_config=semantic_mapping_config,
    ):
        unsupported_terms.append(str(metric_phrase))

    for location_phrase in request_evidence.get("location_phrases", []) or []:
        resolution = _resolve_location_phrase_from_semantic_mapping(
            location_phrase=str(location_phrase),
            semantic_mapping_config=semantic_mapping_config,
        )

        status = resolution.get("status")

        if status in {"resolved", "ignored_dimension"}:
            continue

        if status == "unsupported":
            unsupported_terms.append(str(location_phrase))
            continue

        if status == "ambiguous":
            ambiguous_terms.append(str(location_phrase))
            continue

        if status == "fuzzy":
            suggestion = resolution.get("suggestion", {}) or {}
            suggested_value = suggestion.get("canonical_value")

            ambiguous_terms.append(str(location_phrase))

            if suggested_value:
                evidence_suggestions["location"] = {
                    "original": str(location_phrase),
                    "suggested_value": suggested_value,
                    "suggested_kind": suggestion.get("kind"),
                    "question": f"Did you mean {suggested_value}?",
                }

    unsupported_terms = list(dict.fromkeys(unsupported_terms))
    ambiguous_terms = list(dict.fromkeys(ambiguous_terms))

    if unsupported_terms:
        return {
            "should_continue": False,
            "unsupported_terms": unsupported_terms,
            "ambiguous_terms": ambiguous_terms,
            "evidence_suggestions": evidence_suggestions,
            "error": (
                "Unsupported terms detected: "
                + ", ".join(unsupported_terms)
            ),
        }

    if ambiguous_terms:
        return {
            "should_continue": False,
            "unsupported_terms": [],
            "ambiguous_terms": ambiguous_terms,
            "evidence_suggestions": evidence_suggestions,
            "error": None,
        }

    return {
        "should_continue": True,
        "unsupported_terms": [],
        "ambiguous_terms": [],
        "evidence_suggestions": {},
        "error": None,
    }