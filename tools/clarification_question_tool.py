from typing import Any


def build_clarification_question_tool(
    missing_slots: list[str],
    state_snapshot: dict[str, Any],
) -> str:
    if "valid_mapping" in missing_slots:
        suggestion_question = _build_evidence_suggestion_question(state_snapshot)

        if suggestion_question:
            return suggestion_question

        error = state_snapshot.get("error")

        if error:
            return f"I cannot complete this request because: {error}."

        return (
            "I cannot map part of your request to the available dataset. "
            "Please use the supported dataset fields and metrics."
        )

    if not missing_slots:
        return (
            "I need more information to complete your request. "
            "Please provide additional details."
        )

    if "metric_column" in missing_slots and "time_range" in missing_slots:
        return (
            "I need the metric and the time range to complete your request. "
            "For example: flow for the last 7 days."
        )

    if "metric_column" in missing_slots:
        return (
            "I could not identify which metric you want to query. "
            "Please specify the measurement you need, such as flow or sensor value."
        )

    if "location" in missing_slots:
        return (
            "Which supported city or district should I use for this request?"
        )

    if "time_range" in missing_slots:
        metric = state_snapshot.get("metric_column") or "the selected metric"
        location_hint = _build_location_hint(state_snapshot)

        return (
            f"I need a time range to query {metric}{location_hint}. "
            "For example: last 7 days, last 24 hours, last month, or a specific date range."
        )

    if "chart_type" in missing_slots:
        return (
            "Which chart type would you like to use? "
            "Supported options are line, bar, or table."
        )

    missing_list = ", ".join(missing_slots)

    return (
        "I need more information before I can continue. "
        f"The missing fields are: {missing_list}."
    )


def _build_evidence_suggestion_question(
    state_snapshot: dict[str, Any],
) -> str | None:
    evidence_suggestions = state_snapshot.get("evidence_suggestions") or {}

    if not isinstance(evidence_suggestions, dict):
        return None

    for raw_term, suggestions in evidence_suggestions.items():
        if not raw_term:
            continue

        cleaned_suggestions = _clean_suggestions(suggestions)

        if not cleaned_suggestions:
            continue

        if len(cleaned_suggestions) == 1:
            suggested_value = cleaned_suggestions[0]

            return (
                f"I found a similar supported value for '{raw_term}': "
                f"{suggested_value}. Did you mean {suggested_value}?"
            )

        suggestion_text = ", ".join(cleaned_suggestions)

        return (
            f"I found similar supported values for '{raw_term}': "
            f"{suggestion_text}. Which one did you mean?"
        )

    return None


def _clean_suggestions(suggestions: Any) -> list[str]:
    if not isinstance(suggestions, list):
        return []

    cleaned_suggestions: list[str] = []
    seen = set()

    for suggestion in suggestions:
        if not isinstance(suggestion, str):
            continue

        cleaned_suggestion = suggestion.strip()

        if not cleaned_suggestion:
            continue

        normalized_suggestion = cleaned_suggestion.lower()

        if normalized_suggestion in seen:
            continue

        seen.add(normalized_suggestion)
        cleaned_suggestions.append(cleaned_suggestion)

    return cleaned_suggestions


def _build_location_hint(state_snapshot: dict[str, Any]) -> str:
    filters = state_snapshot.get("filters") or []

    location_parts = []

    for item in filters:
        column = item.get("column")
        value = item.get("value")

        if column in {"city", "district"} and value:
            location_parts.append(f"{column}={value}")

    if not location_parts:
        return ""

    return f" for {' and '.join(location_parts)}"
