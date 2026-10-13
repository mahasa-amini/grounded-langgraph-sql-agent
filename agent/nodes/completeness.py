from typing import cast

from agent.state import SQLAgentState
from agent.utils.debug import debug_node
from tools.required_slots_tool import get_required_slots_tool


def _required_entity_groups(
    requirements_context: dict,
    output_type: str | None,
    intent: str | None,
) -> dict:
    requirements = requirements_context.get("requirements", {})
    key = output_type if output_type in requirements else intent
    if not key or key not in requirements:
        return {}
    groups = requirements[key].get("required_entity_groups", {})
    return groups if isinstance(groups, dict) else {}


def merge_extracted_info_node(state: SQLAgentState) -> SQLAgentState:
    return {}


def check_completeness_node(state: SQLAgentState) -> SQLAgentState:
    requirements_context = state.get("requirements_context") or {}

    required_slots = get_required_slots_tool(
        requirements_context=requirements_context,
        output_type=state.get("output_type"),
        intent=state.get("intent"),
    )

    entity_groups = _required_entity_groups(
        requirements_context,
        state.get("output_type"),
        state.get("intent"),
    )
    managed_slots = {*required_slots, *entity_groups.keys()}
    missing_slots = [
        slot for slot in (state.get("missing_slots", []) or [])
        if slot not in managed_slots
    ]

    slot_values_before_check = {
        slot: state.get(slot)
        for slot in required_slots
    }

    for slot in required_slots:
        value = state.get(slot)

        if value is None:
            missing_slots.append(slot)
            continue

        if isinstance(value, list) and len(value) == 0:
            missing_slots.append(slot)
            continue

        if isinstance(value, dict) and len(value) == 0:
            missing_slots.append(slot)
            continue

        if value == "":
            missing_slots.append(slot)

    available_columns = set(state.get("available_columns", []) or [])
    filters = state.get("filters", []) or []
    for group_name, group_rule in entity_groups.items():
        declared_columns = (
            group_rule.get("any_of", []) if isinstance(group_rule, dict) else []
        )
        supported_columns = {
            column for column in declared_columns if column in available_columns
        }
        has_required_entity = any(
            item.get("column") in supported_columns
            and item.get("value") is not None
            and item.get("value") != ""
            and item.get("value") != []
            for item in filters
        )
        if supported_columns and not has_required_entity:
            missing_slots.append(group_name)

    missing_slots = list(dict.fromkeys(missing_slots))

    is_complete = len(missing_slots) == 0

    request_status = "ready_for_sql" if is_complete else "collecting"

    update = {
        "required_slots": required_slots,
        "missing_slots": missing_slots,
        "is_complete": is_complete,
        "request_status": request_status,
    }

    debug_node(
        "COMPLETENESS CHECK",
        {
            "intent": state.get("intent"),
            "output_type": state.get("output_type"),
            "chart_type": state.get("chart_type"),
            "required_slots": required_slots,
            "required_entity_groups": entity_groups,
            "slot_values_before_check": slot_values_before_check,
            "previous_missing_slots": state.get("missing_slots", []),
            "next_missing_slots": update["missing_slots"],
            "is_complete": update["is_complete"],
            "request_status": update["request_status"],
        },
    )

    return cast(SQLAgentState, update)
