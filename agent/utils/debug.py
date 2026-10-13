import os
from pprint import pformat
from typing import Any


_FALSE_VALUES = {"0", "false", "no", "off"}
_DEFAULT_MAX_CHARS = 3000
_DEFAULT_WIDTH = 120


def is_debug_enabled() -> bool:
    """
    Returns whether agent debug logs should be printed.

    Debug is enabled by default to preserve the current development behavior.
    Set AGENT_DEBUG=false to disable debug logs.
    """
    value = os.getenv("AGENT_DEBUG", "true").strip().lower()
    return value not in _FALSE_VALUES


def get_debug_max_chars() -> int:
    """
    Returns the maximum number of characters printed for each debug value.
    """
    raw_value = os.getenv("AGENT_DEBUG_MAX_CHARS")

    if not raw_value:
        return _DEFAULT_MAX_CHARS

    try:
        max_chars = int(raw_value)
    except ValueError:
        return _DEFAULT_MAX_CHARS

    if max_chars <= 0:
        return _DEFAULT_MAX_CHARS

    return max_chars


def _safe_format_value(value: Any) -> str:
    """
    Safely formats a value for debug output.
    """
    try:
        return pformat(
            value,
            width=_DEFAULT_WIDTH,
            compact=False,
            sort_dicts=False,
        )
    except Exception:
        try:
            return repr(value)
        except Exception:
            return "<unprintable value>"


def _truncate_text(text: str, max_chars: int) -> str:
    """
    Truncates long debug values to keep logs readable.
    """
    if len(text) <= max_chars:
        return text

    truncated_count = len(text) - max_chars
    return f"{text[:max_chars]}\n... <truncated {truncated_count} characters>"


def _indent_multiline_text(text: str, spaces: int = 2) -> str:
    """
    Indents multi-line values under their debug key.
    """
    prefix = " " * spaces
    return "\n".join(f"{prefix}{line}" for line in text.splitlines())


def debug_node(title: str, values: dict[str, Any]) -> None:
    """
    Prints a standardized debug block for a LangGraph node.

    Example output:

    ==================== NODE: EXAMPLE ====================
    key: 'value'
    nested:
      {'a': 1, 'b': 2}
    =======================================================
    """
    if not is_debug_enabled():
        return

    normalized_title = title.strip().upper()
    header = f" NODE: {normalized_title} "
    separator = "=" * max(len(header), 60)

    max_chars = get_debug_max_chars()

    print("\n" + separator)
    print(header.center(len(separator), "="))

    for key, value in values.items():
        formatted_value = _safe_format_value(value)
        formatted_value = _truncate_text(formatted_value, max_chars)

        if "\n" in formatted_value:
            print(f"{key}:")
            print(_indent_multiline_text(formatted_value))
        else:
            print(f"{key}: {formatted_value}")

    print(separator + "\n")