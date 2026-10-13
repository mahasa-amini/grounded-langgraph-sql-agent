from agent.prompts.schema_mapping_prompt import SCHEMA_MAPPING_PROMPT
from models.llm import get_mapping_llm
from models.structured_outputs import SchemaMappingOutput


def get_schema_mapping_chain():
    llm = get_mapping_llm().with_structured_output(SchemaMappingOutput)

    return SCHEMA_MAPPING_PROMPT | llm