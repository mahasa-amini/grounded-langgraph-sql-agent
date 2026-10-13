from agent.prompts.sql_generation_prompt import SQL_GENERATION_PROMPT
from models.llm import get_sql_llm
from models.structured_outputs import SQLGenerationOutput


def get_sql_generation_chain():
    llm = get_sql_llm().with_structured_output(
        SQLGenerationOutput,
        include_raw=True,
    )

    return SQL_GENERATION_PROMPT | llm
