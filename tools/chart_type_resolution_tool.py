import re
from typing import Any


_ALLOWED_CHART_TYPES = {"line", "bar"}

_BAR_CHART_PATTERNS = [
    re.compile(r"\bbar\s+chart\b", flags=re.IGNORECASE),
    re.compile(r"\bbar\s+graph\b", flags=re.IGNORECASE),
    re.compile(r"\bcolumn\s+chart\b", flags=re.IGNORECASE),
]

_LINE_CHART_PATTERNS = [
    re.compile(r"\bline\s+chart\b", flags=re.IGNORECASE),
    re.compile(r"\bline\s+graph\b", flags=re.IGNORECASE),
    re.compile(r"\bline\s+plot\b", flags=re.IGNORECASE),
    re.compile(r"\btrend\s+chart\b", flags=re.IGNORECASE),
    re.compile(r"\btime\s+series\s+chart\b", flags=re.IGNORECASE),
]


def resolve_chart_type_from_text(
    chart_phrase: str | None,
    request_text: str | None,
    current_chart_type: str | None,
) -> dict[str, Any]:
    """
    Deterministically resolve chart type from explicit user wording.

    The explicit chart phrase/request text must override mapper output or
    previous state. This prevents cases such as:
    - "bar chart" becoming "line"
    - chart_type leaking from a previous request
    """

    evidence_text = _normalize_text(chart_phrase)
    full_request_text = _normalize_text(request_text)
    combined_text = f"{evidence_text} {full_request_text}".strip()

    explicit_chart_type = _detect_explicit_chart_type(combined_text)

    if explicit_chart_type:
        return {
            "chart_type": explicit_chart_type,
            "source": "explicit_chart_text",
            "changed": explicit_chart_type != current_chart_type,
        }

    normalized_current_chart_type = _normalize_chart_type(current_chart_type)

    return {
        "chart_type": normalized_current_chart_type,
        "source": "current_chart_type",
        "changed": False,
    }


def _detect_explicit_chart_type(text: str) -> str | None:
    if not text:
        return None

    if any(pattern.search(text) for pattern in _BAR_CHART_PATTERNS):
        return "bar"

    if any(pattern.search(text) for pattern in _LINE_CHART_PATTERNS):
        return "line"

    return None


def _normalize_chart_type(chart_type: str | None) -> str | None:
    if not isinstance(chart_type, str):
        return None

    normalized_chart_type = chart_type.strip().lower()

    if normalized_chart_type in _ALLOWED_CHART_TYPES:
        return normalized_chart_type

    return None


def _normalize_text(text: str | None) -> str:
    if not isinstance(text, str):
        return ""

    return " ".join(text.strip().split())