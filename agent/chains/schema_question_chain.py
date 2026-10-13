from agent.prompts.schema_question_prompt import SCHEMA_QUESTION_PROMPT
from models.llm import get_answer_llm
from models.structured_outputs import SchemaQuestionAnswerOutput


def get_schema_question_chain():
    llm = get_answer_llm().with_structured_output(SchemaQuestionAnswerOutput)

    return SCHEMA_QUESTION_PROMPT | llm