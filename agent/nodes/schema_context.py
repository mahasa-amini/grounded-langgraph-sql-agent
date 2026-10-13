from typing import cast
from agent.state import SQLAgentState
from app.config import settings
from tools.schema_context_tool import get_schema_context_tool




def load_schema_context_node(state: SQLAgentState) -> SQLAgentState:
    schema_data = get_schema_context_tool()

    schema_context = schema_data["schema_context"]
    semantic_context = schema_data["semantic_context"]
    requirements_context = schema_data["requirements_context"]
    public_schema_context = schema_data["public_schema_context"]

    table_name = requirements_context.get("defaults", {}).get(
        "table",
        "sensor_measurements",
    )

    table_data = schema_context.get("tables", {}).get(table_name, {})
    columns = table_data.get("columns", {})

    return {
        "schema_context": schema_context,
        "semantic_context": semantic_context,
        "requirements_context": requirements_context,
        "table_name": table_name,
        "available_columns": list(columns.keys()),
        "max_clarification_count": settings.max_clarification_count,
        "public_schema_context": public_schema_context,
    }
