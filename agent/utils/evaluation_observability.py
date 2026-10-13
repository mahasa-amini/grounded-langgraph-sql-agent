from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator


_capture: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "evaluation_observability_capture", default=None
)


def is_evaluation_observability_enabled() -> bool:
    return _capture.get() is not None


def record_evaluation_event(stage: str, payload: dict[str, Any]) -> None:
    events = _capture.get()
    if events is not None:
        events.append({"stage": stage, **payload})


@contextmanager
def capture_evaluation_observability() -> Iterator[list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    token = _capture.set(events)
    try:
        yield events
    finally:
        _capture.reset(token)


def compact_model_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return {str(key): compact_model_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [compact_model_value(item) for item in value]
    content = getattr(value, "content", None)
    if content is not None:
        return {
            "type": getattr(value, "type", type(value).__name__),
            "content": content,
            "tool_calls": getattr(value, "tool_calls", None),
            "additional_kwargs": getattr(value, "additional_kwargs", None),
        }
    return str(value)
