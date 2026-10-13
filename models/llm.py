from functools import lru_cache
from typing import Literal

from langchain_ollama import ChatOllama

from app.config import settings


LLMRole = Literal[
    "interpret",
    "mapping",
    "clarification",
    "sql",
    "answer",
]


def _create_ollama_llm(model: str, temperature: float) -> ChatOllama:
    """
    Create a ChatOllama instance with the given model and temperature.
    """

    return ChatOllama(
        model=model,
        base_url=settings.ollama_base_url,
        temperature=temperature,

    )


@lru_cache(maxsize=1)
def get_interpret_llm() -> ChatOllama:
    """
    LLM used for understanding the user's natural language request.
    """

    return _create_ollama_llm(
        model=settings.interpret_model,
        temperature=settings.interpret_temperature,
    )


@lru_cache(maxsize=1)
def get_mapping_llm() -> ChatOllama:
    """
    LLM used for mapping user terms to database schema concepts.
    """

    return _create_ollama_llm(
        model=settings.mapping_model,
        temperature=settings.mapping_temperature,
    )


@lru_cache(maxsize=1)
def get_clarification_llm() -> ChatOllama:
    """
    LLM used for generating clarification questions.
    """

    return _create_ollama_llm(
        model=settings.clarification_model,
        temperature=settings.clarification_temperature,
    )


@lru_cache(maxsize=1)
def get_sql_llm() -> ChatOllama:
    """
    LLM used for generating SQL queries.
    """

    return _create_ollama_llm(
        model=settings.sql_model,
        temperature=settings.sql_temperature,
    )


@lru_cache(maxsize=1)
def get_answer_llm() -> ChatOllama:
    """
    LLM used for formatting the final answer for the user.
    """

    return _create_ollama_llm(
        model=settings.answer_model,
        temperature=settings.answer_temperature,
    )


def get_llm_by_role(role: LLMRole) -> ChatOllama:
    """
    Return the right LLM based on the role name.
    """

    if role == "interpret":
        return get_interpret_llm()

    if role == "mapping":
        return get_mapping_llm()

    if role == "clarification":
        return get_clarification_llm()

    if role == "sql":
        return get_sql_llm()

    if role == "answer":
        return get_answer_llm()

    raise ValueError(f"Unknown LLM role: {role}")