from agent.state import TimeRange


_VALID_TIME_RANGE_KINDS = {"relative", "named", "absolute"}


def resolve_effective_time_range(
    parsed_time_range: TimeRange | None,
    previous_time_range: TimeRange | None,
    time_phrase: object,
) -> TimeRange | None:
    """
    Resolves the effective time range for the current request.

    Rules:
    - If deterministic parsing produced a parsed_time_range, use it as the source of truth.
    - If the user provided a new time phrase but parsing failed, return None so clarification can happen.
    - If there is no new time phrase, keep a valid previous time range.
    - Otherwise, return None.
    """
    if parsed_time_range is not None:
        return parsed_time_range

    has_time_phrase = isinstance(time_phrase, str) and bool(time_phrase.strip())

    if has_time_phrase:
        return None

    if (
        isinstance(previous_time_range, dict)
        and previous_time_range.get("kind") in _VALID_TIME_RANGE_KINDS
    ):
        return previous_time_range

    return None