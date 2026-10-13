from typing import Any, Literal, Optional
from typing_extensions import TypedDict, Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


OutputType = Literal["text", "chart"]
ChartType = Literal["line", "bar", "table"]
IntentType = Literal["retrieve_data", "unknown"]

UserMessageType = Literal[
    "general_chat",
    "schema_question",
    "data_request",
    "chart_request",
    "clarification_or_update",
    "goodbye",
    "unclear",
    "reset_request",
]

ActiveRequestType = Literal["data", "chart"]

ActiveRequestAction = Literal[
    "none",
    "invalid",
    "clarification_answer",
    "correction",
    "side_question",
    "new_request",
    "cancel",
    "ambiguous",
    "continue_ambiguous",
    "new_request_from_ambiguous",
]

RequestStatus = Literal[
    "idle",
    "collecting",
    "ready_for_sql",
]

ConversationStatus = Literal[
    "active",
    "ended",
]


class TimeRange(TypedDict, total=False):
    # Backward-compatible fields
    start: Optional[str]
    end: Optional[str]
    relative: Optional[str]

    # Canonical parsed time range fields
    kind: Literal["relative", "named", "absolute"]
    amount: int
    unit: Literal["SECOND", "MINUTE", "HOUR", "DAY"]
    name: Literal["today", "yesterday"]


class QueryFilter(TypedDict, total=False):
    column: str
    operator: str
    value: Any

class QueryCondition(QueryFilter, total=False):
    source: str

class RequestEvidence(TypedDict, total=False):
    metric_phrase: Optional[str]
    location_phrases: list[str]
    time_phrase: Optional[str]
    aggregation_phrase: Optional[str]
    chart_phrase: Optional[str]
    aggregation_conflict: Optional[str]
    time_phrase_provenance: Optional[str]
    time_fallback_result: dict[str, Any]


class SQLAgentState(TypedDict, total=False):
    # Conversation
    messages: Annotated[list[BaseMessage], add_messages]
    latest_user_message: Optional[str]
    user_message_type: Optional[UserMessageType]
    classification_reason: Optional[str]
    active_request_action: Optional[ActiveRequestAction]
    correction_targets: list[str]
    classified_request_type: Optional[ActiveRequestType]
    classification_valid: bool
    classification_error: Optional[str]
    model_classification: Optional[dict[str, Any]]
    classification_safeguard: Optional[str]
    request_transition_valid: bool
    conversation_status: Optional[ConversationStatus]

    # Active request tracking
    request_status: Optional[RequestStatus]
    active_request_type: Optional[ActiveRequestType]
    active_request_text: Optional[str]
    clarification_answers: list[str]
    mapping_input_text: Optional[str]
    pending_ambiguous_message: Optional[str]
    pending_ambiguous_request_type: Optional[ActiveRequestType]
    pending_ambiguous_correction_targets: list[str]

    # Request evidence
    request_evidence: Optional[RequestEvidence]
    evidence_suggestions: dict[str, Any]

    # Schema context
    schema_context: Optional[dict[str, Any]]
    semantic_context: Optional[dict[str, Any]]
    requirements_context: Optional[dict[str, Any]]
    public_schema_context: Optional[dict[str, Any]]

    # Table context - currently focused on one table
    table_name: Optional[str]
    available_columns: list[str]

    # User intent and requested output
    intent: Optional[IntentType]
    output_type: Optional[OutputType]
    chart_type: Optional[ChartType]

    # Raw interpretation
    raw_metric: Optional[str]

    # Mapped query fields
    selected_columns: list[str]
    metric_column: Optional[str]
    time_column: Optional[str]
    time_range: Optional[TimeRange]
    parsed_time_range: Optional[TimeRange]
    filters: list[QueryFilter]
    aggregation: Optional[str]
    group_by: list[str]
    order_by: Optional[str]
    limit: Optional[int]
    unsupported_terms: list[str]
    ambiguous_terms: list[str]
    having_conditions: list[QueryCondition]
    distinct: bool

    # Completeness tracking
    required_slots: list[str]
    missing_slots: list[str]
    is_complete: bool

    # Clarification loop
    clarification_question: Optional[str]
    clarification_count: int
    max_clarification_count: int

    # SQL pipeline
    sql_query: Optional[str]
    sql_is_safe: bool
    sql_validation_errors: list[str]
    sql_explanation: Optional[str]
    sql_generation_provenance: Optional[str]
    sql_compilation_errors: list[str]

    # Execution and response
    query_result: Optional[Any]
    execution_error: Optional[str]
    final_answer: Optional[str]
    chart_payload: Optional[dict[str, Any]]

    # General error/debug
    error: Optional[str]
