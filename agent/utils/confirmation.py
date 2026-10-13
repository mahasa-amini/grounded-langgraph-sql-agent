from typing import Any
import re

from agent.utils.message_text import extract_plain_text_from_message_like_value


def normalize_confirmation_message(message: str) -> str:
    punctuation_to_space = ",.!?;:"

    normalized_message = message.lower().strip()

    for punctuation in punctuation_to_space:
        normalized_message = normalized_message.replace(punctuation, " ")

    return " ".join(normalized_message.split())


def normalize_text_for_affirmative_confirmation(value: Any) -> str:
    return " ".join(
        re.sub(
            r"[^a-zA-Z0-9]+",
            " ",
            str(value).lower(),
        )
        .strip()
        .split()
    )


def is_simple_confirmation_message(message: str) -> bool:
    normalized_message = normalize_confirmation_message(message)

    yes_like_messages = {
        "yes",
        "yeah",
        "yep",
        "correct",
        "right",
        "that's right",
        "that is right",
        "ok",
        "okay",
        "sure",
        "yes please",
    }

    return normalized_message in yes_like_messages


def is_affirmative_confirmation_text(value: Any) -> bool:
    normalized_text = normalize_text_for_affirmative_confirmation(
        extract_plain_text_from_message_like_value(value)
    )

    if not normalized_text:
        return False

    affirmative_values = {
        "yes",
        "yeah",
        "yep",
        "yup",
        "correct",
        "right",
        "exactly",
        "ok",
        "okay",
        "sure",
        "confirmed",
        "confirm",
        "that is right",
        "thats right",
    }

    return normalized_text in affirmative_values


def get_single_evidence_suggestion(
    evidence_suggestions: dict[str, list[str]] | None,
) -> str | None:
    if not isinstance(evidence_suggestions, dict):
        return None

    if len(evidence_suggestions) != 1:
        return None

    suggestions = next(iter(evidence_suggestions.values()))

    if not isinstance(suggestions, list):
        return None

    cleaned_suggestions = [
        suggestion.strip()
        for suggestion in suggestions
        if isinstance(suggestion, str) and suggestion.strip()
    ]

    unique_suggestions = []
    seen = set()

    for suggestion in cleaned_suggestions:
        normalized_suggestion = suggestion.lower()

        if normalized_suggestion in seen:
            continue

        seen.add(normalized_suggestion)
        unique_suggestions.append(suggestion)

    if len(unique_suggestions) != 1:
        return None

    return unique_suggestions[0]


def resolve_confirmation_answer_from_suggestion(
    latest_user_message: str,
    evidence_suggestions: dict[str, list[str]] | None,
) -> str:
    if not is_simple_confirmation_message(latest_user_message):
        return latest_user_message

    suggested_value = get_single_evidence_suggestion(evidence_suggestions)

    if not suggested_value:
        return latest_user_message

    return suggested_value


def replace_confirmed_location_phrase_in_text(
    text: Any,
    original_location: Any,
    suggested_location: Any,
) -> str:
    text_value = str(text or "")
    original_value = str(original_location or "").strip()
    suggested_value = str(suggested_location or "").strip()

    if not text_value or not original_value or not suggested_value:
        return text_value

    pattern = rf"(?<![a-zA-Z0-9_]){re.escape(original_value)}(?![a-zA-Z0-9_])"

    return re.sub(
        pattern=pattern,
        repl=suggested_value,
        string=text_value,
        flags=re.IGNORECASE,
    )


def try_apply_affirmative_location_confirmation(
    latest_user_message: Any,
    active_request_text: Any,
    missing_slots: list[str],
    evidence_suggestions: dict[str, Any],
) -> dict[str, Any]:
    if "location_confirmation" not in (missing_slots or []):
        return {
            "applied": False,
            "active_request_text": active_request_text,
            "resolved_clarification_answer": latest_user_message,
            "missing_slots": missing_slots,
            "evidence_suggestions": evidence_suggestions,
        }

    if not is_affirmative_confirmation_text(latest_user_message):
        return {
            "applied": False,
            "active_request_text": active_request_text,
            "resolved_clarification_answer": latest_user_message,
            "missing_slots": missing_slots,
            "evidence_suggestions": evidence_suggestions,
        }

    location_suggestion = (evidence_suggestions or {}).get("location", {}) or {}
    original_location = location_suggestion.get("original")
    suggested_location = location_suggestion.get("suggested_value")

    if not original_location or not suggested_location:
        return {
            "applied": False,
            "active_request_text": active_request_text,
            "resolved_clarification_answer": latest_user_message,
            "missing_slots": missing_slots,
            "evidence_suggestions": evidence_suggestions,
        }

    updated_active_request_text = replace_confirmed_location_phrase_in_text(
        text=active_request_text,
        original_location=original_location,
        suggested_location=suggested_location,
    )

    updated_missing_slots = [
        slot
        for slot in (missing_slots or [])
        if slot != "location_confirmation"
    ]

    return {
        "applied": True,
        "active_request_text": updated_active_request_text,
        "resolved_clarification_answer": str(suggested_location),
        "missing_slots": updated_missing_slots,
        "evidence_suggestions": {},
    }