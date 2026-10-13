from __future__ import annotations

from datetime import datetime
from numbers import Real
from typing import Any, Iterable, Sequence, TypeVar


SchemaFieldT = TypeVar("SchemaFieldT")
REQUIRED_PREVIEW_FIELDS = frozenset({"district", "timestamp", "value"})


def select_preview_fields_in_physical_order(
    physical_schema: Sequence[SchemaFieldT],
    required_names: Iterable[str] = REQUIRED_PREVIEW_FIELDS,
) -> list[SchemaFieldT]:
    """Filter required preview fields without changing physical schema order."""
    required = set(required_names)
    selected = [
        field
        for field in physical_schema
        if getattr(field, "name", None) in required
    ]
    selected_names = [getattr(field, "name", None) for field in selected]
    missing = required.difference(selected_names)
    duplicates = {
        name for name in selected_names if selected_names.count(name) > 1
    }
    if missing or duplicates:
        details = []
        if missing:
            details.append("missing=" + ",".join(sorted(missing)))
        if duplicates:
            details.append("duplicates=" + ",".join(sorted(duplicates)))
        raise ValueError("Invalid preview schema: " + "; ".join(details))
    return selected


def decode_preview_row_by_name(row: Any) -> dict[str, Any]:
    """Read typed preview values by field name, never by positional index."""
    decoded = {
        "district": row["district"],
        "timestamp": row["timestamp"],
        "value": row["value"],
    }
    if not isinstance(decoded["district"], str):
        raise ValueError("Preview district did not decode as text")
    if not isinstance(decoded["timestamp"], datetime):
        raise ValueError("Preview timestamp did not decode as datetime")
    if decoded["value"] is not None and (
        isinstance(decoded["value"], bool) or not isinstance(decoded["value"], Real)
    ):
        raise ValueError("Preview value did not decode as numeric")
    return decoded
