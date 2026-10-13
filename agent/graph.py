from langgraph.graph import END, START, StateGraph

from agent.nodes.clarification import (
    answer_clarification_node,
    check_clarification_limit_node,
    fail_gracefully_node,
)
from agent.nodes.completeness import (
    check_completeness_node,
    merge_extracted_info_node,
)
from agent.nodes.conversation import (
    answer_goodbye_node,
    wait_for_next_message_node,
)
from agent.nodes.sql_execution import execute_sql_node
from agent.nodes.response_builder import build_text_response_node
from agent.nodes.general_chat import answer_general_chat_node
from agent.nodes.mapping_validation import validate_mapped_state_node
from agent.nodes.message_classification import classify_user_message_node
from agent.nodes.message_input import receive_user_message_node
from agent.nodes.request_evidence import extract_request_evidence_node
from agent.nodes.request_interpretation import interpret_user_request_node
from agent.nodes.request_lifecycle import (
    mark_request_done_node,
    reset_active_request_node,
    start_or_update_active_request_node,
)
from agent.nodes.request_relationship import ask_request_relationship_node
from agent.nodes.request_validation import (
    pre_validate_request_evidence_node,
    route_after_pre_validate_request_evidence,
)
from agent.nodes.chart_builder import build_chart_payload_node
from agent.nodes.schema_context import load_schema_context_node
from agent.nodes.schema_mapping import map_to_schema_node
from agent.nodes.schema_question import answer_schema_question_node
from agent.nodes.sql_generation import generate_sql_node
from agent.nodes.sql_safety import (
    early_sql_safety_check_node,
    security_reject_message_node,
    validate_sql_safety_node,
)
from agent.nodes.time_parsing import parse_time_phrase_node
from agent.routers.clarification_router import route_clarification_limit
from agent.routers.completeness_router import route_completeness
from agent.routers.message_router import route_after_early_sql_safety_check
from agent.routers.sql_safety_router import route_sql_safety
from agent.routers.output_router import route_after_sql_execution
from agent.state import SQLAgentState
from memory.checkpointer import checkpointer


def route_after_request_lifecycle(state: SQLAgentState) -> str:
    return "interpret" if state.get("request_transition_valid") is True else "relationship"


def build_graph(use_checkpointer: bool = True):
    graph_builder = StateGraph(SQLAgentState)

    # -------------------------------------------------------------------------
    # Node registration
    # -------------------------------------------------------------------------

    # Initial message processing
    graph_builder.add_node("load_schema_context", load_schema_context_node)
    graph_builder.add_node("receive_user_message", receive_user_message_node)
    graph_builder.add_node("classify_user_message", classify_user_message_node)
    graph_builder.add_node("early_sql_safety_check", early_sql_safety_check_node)

    # Simple conversational paths
    graph_builder.add_node("answer_general_chat", answer_general_chat_node)
    graph_builder.add_node("answer_schema_question", answer_schema_question_node)
    graph_builder.add_node("answer_goodbye", answer_goodbye_node)
    graph_builder.add_node("security_reject_message", security_reject_message_node)
    graph_builder.add_node("wait_for_next_message", wait_for_next_message_node)
    graph_builder.add_node("ask_request_relationship", ask_request_relationship_node)

    # Request lifecycle
    graph_builder.add_node("start_or_update_active_request", start_or_update_active_request_node)
    graph_builder.add_node("reset_active_request", reset_active_request_node)
    graph_builder.add_node("mark_request_done", mark_request_done_node)

    # Request interpretation and evidence extraction
    graph_builder.add_node("interpret_user_request", interpret_user_request_node)
    graph_builder.add_node("extract_request_evidence", extract_request_evidence_node)
    graph_builder.add_node("parse_time_phrase", parse_time_phrase_node)
    graph_builder.add_node("pre_validate_request_evidence", pre_validate_request_evidence_node)

    # Schema mapping and validation
    graph_builder.add_node("map_to_schema", map_to_schema_node)
    graph_builder.add_node("validate_mapped_state", validate_mapped_state_node)
    graph_builder.add_node("merge_extracted_info", merge_extracted_info_node)
    graph_builder.add_node("check_completeness", check_completeness_node)

    # Clarification flow
    graph_builder.add_node("check_clarification_limit", check_clarification_limit_node)
    graph_builder.add_node("answer_clarification", answer_clarification_node)
    graph_builder.add_node("fail_gracefully", fail_gracefully_node)

    # SQL generation and validation
    graph_builder.add_node("generate_sql", generate_sql_node)
    graph_builder.add_node("validate_sql_safety", validate_sql_safety_node)
    graph_builder.add_node("execute_sql", execute_sql_node)
    graph_builder.add_node("build_text_response", build_text_response_node)
    graph_builder.add_node("build_chart_payload", build_chart_payload_node)

    # -------------------------------------------------------------------------
    # Main entry path
    # -------------------------------------------------------------------------

    graph_builder.add_edge(START, "load_schema_context")
    graph_builder.add_edge("load_schema_context", "receive_user_message")
    graph_builder.add_edge("receive_user_message", "classify_user_message")
    graph_builder.add_edge("classify_user_message", "early_sql_safety_check")

    # -------------------------------------------------------------------------
    # Top-level routing after message classification and early SQL safety
    # -------------------------------------------------------------------------

    graph_builder.add_conditional_edges(
        "early_sql_safety_check",
        route_after_early_sql_safety_check,
        {
            "security_reject": "security_reject_message",
            "general_chat": "answer_general_chat",
            "schema_question": "answer_schema_question",
            "goodbye": "answer_goodbye",
            "data_or_chart_request": "start_or_update_active_request",
            "clarification_or_update": "start_or_update_active_request",
            "reset_request": "reset_active_request",
            "unclear": "answer_general_chat",
            "ambiguous_request_relationship": "ask_request_relationship",
        },
    )

    # -------------------------------------------------------------------------
    # Terminal paths for simple responses
    # -------------------------------------------------------------------------

    graph_builder.add_edge("answer_general_chat", "wait_for_next_message")
    graph_builder.add_edge("answer_schema_question", "wait_for_next_message")
    graph_builder.add_edge("ask_request_relationship", "wait_for_next_message")
    graph_builder.add_edge("security_reject_message", "wait_for_next_message")
    graph_builder.add_edge("reset_active_request", "wait_for_next_message")
    graph_builder.add_edge("answer_goodbye", END)
    graph_builder.add_edge("wait_for_next_message", END)

    # -------------------------------------------------------------------------
    # Data/chart request processing path
    # -------------------------------------------------------------------------

    graph_builder.add_conditional_edges(
        "start_or_update_active_request",
        route_after_request_lifecycle,
        {
            "interpret": "interpret_user_request",
            "relationship": "ask_request_relationship",
        },
    )
    graph_builder.add_edge("interpret_user_request", "extract_request_evidence")
    graph_builder.add_edge("extract_request_evidence", "parse_time_phrase")
    graph_builder.add_edge("parse_time_phrase", "pre_validate_request_evidence")

    graph_builder.add_conditional_edges(
        "pre_validate_request_evidence",
        route_after_pre_validate_request_evidence,
        {
            "continue_to_mapping": "map_to_schema",
            "stop_for_clarification": "answer_clarification",
        },
    )

    graph_builder.add_edge("map_to_schema", "validate_mapped_state")
    graph_builder.add_edge("validate_mapped_state", "merge_extracted_info")
    graph_builder.add_edge("merge_extracted_info", "check_completeness")

    # -------------------------------------------------------------------------
    # Completeness and clarification routing
    # -------------------------------------------------------------------------

    graph_builder.add_conditional_edges(
        "check_completeness",
        route_completeness,
        {
            "complete": "generate_sql",
            "incomplete": "check_clarification_limit",
        },
    )

    graph_builder.add_conditional_edges(
        "check_clarification_limit",
        route_clarification_limit,
        {
            "can_continue": "answer_clarification",
            "limit_exceeded": "fail_gracefully",
        },
    )

    graph_builder.add_edge("answer_clarification", "wait_for_next_message")
    graph_builder.add_edge("fail_gracefully", "wait_for_next_message")

    # -------------------------------------------------------------------------
    # SQL generation, safety validation, and completion
    # -------------------------------------------------------------------------

    graph_builder.add_edge("generate_sql", "validate_sql_safety")

    graph_builder.add_conditional_edges(
        "validate_sql_safety",
        route_sql_safety,
        {
            "safe": "execute_sql",
            "unsafe": "security_reject_message",
        },
    )

    graph_builder.add_conditional_edges(
    "execute_sql",
    route_after_sql_execution,
    {
        "text": "build_text_response",
        "chart": "build_chart_payload",
    },
)

    graph_builder.add_edge("build_text_response", "mark_request_done")
    graph_builder.add_edge("build_chart_payload", "mark_request_done")
    graph_builder.add_edge("mark_request_done", "wait_for_next_message")

    # -------------------------------------------------------------------------
    # Compile graph
    # -------------------------------------------------------------------------

    if use_checkpointer:
        return graph_builder.compile(checkpointer=checkpointer)

    return graph_builder.compile()


app = build_graph(use_checkpointer=True)
server_app = build_graph(use_checkpointer=False)
