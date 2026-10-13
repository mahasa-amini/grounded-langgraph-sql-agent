from typing import Any


def get_required_slots_tool(
    requirements_context: dict[str, Any],
    output_type: str | None,
    intent: str | None,
) -> list[str]:
    requirements = requirements_context.get("requirements", {})

    if output_type and output_type in requirements:
        return requirements[output_type].get("required_slots", [])

    if intent and intent in requirements:
        return requirements[intent].get("required_slots", [])

    return []