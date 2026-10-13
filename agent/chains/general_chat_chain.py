from agent.prompts.general_chat_prompt import GENERAL_CHAT_PROMPT
from models.llm import get_answer_llm
from models.structured_outputs import GeneralChatOutput


def get_general_chat_chain():
    llm = get_answer_llm().with_structured_output(GeneralChatOutput)

    return GENERAL_CHAT_PROMPT | llm