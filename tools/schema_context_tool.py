from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATABASE_SCHEMA_PATH = PROJECT_ROOT / "schema" / "database_schema.yaml"
SEMANTIC_MAPPING_PATH = PROJECT_ROOT / "schema" / "semantic_mapping.yaml"
REQUIREMENTS_PATH = PROJECT_ROOT / "schema" / "requirements.yaml"
PUBLIC_SCHEMA_PATH = PROJECT_ROOT / "schema" / "public_schema.yaml"


def _load_yaml_file(file_path: Path) -> dict[str, Any]:
    if not file_path.exists():
        raise FileNotFoundError(f"Schema file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML file to contain a dictionary: {file_path}")

    return data


def get_schema_context_tool() -> dict[str, Any]:
    database_schema = _load_yaml_file(DATABASE_SCHEMA_PATH)
    semantic_mapping = _load_yaml_file(SEMANTIC_MAPPING_PATH)
    requirements = _load_yaml_file(REQUIREMENTS_PATH)
    public_schema = _load_yaml_file(PUBLIC_SCHEMA_PATH)

    return {
        "schema_context": database_schema,
        "semantic_context": semantic_mapping,
        "requirements_context": requirements,
        "public_schema_context": public_schema,
    }