from agent.prompts.message_classification_prompt import MESSAGE_CLASSIFICATION_PROMPT
from models.llm import get_interpret_llm
from models.structured_outputs import MessageClassificationOutput


def get_message_classification_chain(include_raw: bool = False):
    llm = get_interpret_llm().with_structured_output(
        MessageClassificationOutput,
        include_raw=include_raw,
    )

    return MESSAGE_CLASSIFICATION_PROMPT | llm
