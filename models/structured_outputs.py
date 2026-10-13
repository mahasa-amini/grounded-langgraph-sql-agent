from pydantic import BaseModel, Field, field_validator
from typing import Literal


UserMessageTypeOutput = Literal[
    "general_chat",
    "schema_question",
    "data_request",
    "chart_request",
    "clarification_or_update",
    "goodbye",
    "unclear",
    "reset_request",
]

ActiveRequestActionOutput = Literal[
    "none",
    "clarification_answer",
    "correction",
    "side_question",
    "new_request",
    "cancel",
    "ambiguous",
    "continue_ambiguous",
    "new_request_from_ambiguous",
]

CorrectionTargetOutput = Literal[
    "metric",
    "location",
    "time_range",
    "aggregation",
    "chart_type",
    "output_type",
    "group_by",
    "order_by",
    "limit",
    "filters",
]


class SchemaQuestionAnswerOutput(BaseModel):
    answer: str = Field(
        description="A concise user-facing answer to the schema question."
    )


class GeneralChatOutput(BaseModel):
    answer: str = Field(
        description="A helpful conversational answer to the user's general message."
    )


class SQLGenerationOutput(BaseModel):
    sql_query: str = Field(
        description="A single safe read-only SELECT SQL query."
    )

    explanation: str | None = Field(
        default=None,
        description="A short explanation of what the SQL query does.",
    )


class MappedFilterOutput(BaseModel):
    column: str = Field(
        description="The schema column used for filtering."
    )

    operator: str = Field(
        description="The filter operator, such as =, !=, >, >=, <, <=, IN, or BETWEEN."
    )

    value: str | int | float | list[str] = Field(
        description="The filter value mapped from the user request."
    )


class TimeRangeOutput(BaseModel):
    """
    Deprecated for schema mapping.

    Time ranges are handled by the deterministic parser, not by SchemaMappingOutput.
    This class is kept only for backward compatibility in case other modules still import it.
    """

    start: str | None = Field(
        default=None,
        description="Deprecated. Do not use in schema mapping."
    )

    end: str | None = Field(
        default=None,
        description="Deprecated. Do not use in schema mapping."
    )

    relative: str | None = Field(
        default=None,
        description="Deprecated. Do not use in schema mapping."
    )


class RequestEvidenceOutput(BaseModel):
    metric_phrase: str | None = Field(
        default=None,
        description=(
            "The exact metric or measurement phrase explicitly requested by the user. "
            "For example: flow, water flow, pressure, pressure value, temperature. "
            "Return null if the user did not explicitly name a measured quantity."
        ),
    )

    location_phrases: list[str] = Field(
        default_factory=list,
        description=(
            "Exact location-like phrases explicitly requested by the user. "
            "For example: Marconi, Piemonte, Marene. "
            "Do not validate them against the schema here."
        ),
    )

    time_phrase: str | None = Field(
        default=None,
        description=(
            "The exact time expression explicitly requested by the user. "
            "For example: last 7 days, last 7d, yesterday, since Monday."
        ),
    )

    aggregation_phrase: str | None = Field(
        default=None,
        description=(
            "The exact aggregation phrase explicitly requested by the user. "
            "For example: maximum, minimum, average, total, count."
        ),
    )

    chart_phrase: str | None = Field(
        default=None,
        description=(
            "The exact chart or visualization phrase explicitly requested by the user. "
            "For example: chart, line chart, linear chart, bar chart, graph."
        ),
    )


class SchemaMappingOutput(BaseModel):
    table_name: str | None = Field(
        default=None,
        description="The mapped database table name."
    )

    active_request_type: Literal["data", "chart"] | None = Field(
        default=None,
        description="The active request type."
    )

    output_type: Literal["text", "chart"] | None = Field(
        default=None,
        description="The requested output type."
    )

    chart_type: Literal["line", "bar", "table"] | None = Field(
        default=None,
        description="The requested chart type if the user wants a chart."
    )

    metric_column: str | None = Field(
        default=None,
        description="The mapped metric column."
    )

    raw_metric: str | None = Field(
        default=None,
        description=(
            "The metric or measurement concept explicitly requested by the user "
            "before schema mapping."
        ),
    )

    time_column: str | None = Field(
        default=None,
        description="The mapped time column."
    )

    time_range: None = Field(
        default=None,
        description=(
            "Always null for schema mapping. "
            "Time parsing is handled by a deterministic Python parser outside this mapper. "
            "Do not generate start, end, or relative time values."
        ),
    )

    filters: list[MappedFilterOutput] = Field(
        default_factory=list,
        description="Filters mapped from the user request."
    )

    selected_columns: list[str] = Field(
        default_factory=list,
        description="Columns to select in SQL."
    )

    aggregation: str | None = Field(
        default=None,
        description="Requested aggregation such as avg, sum, min, max, or count."
    )

    group_by: list[str] = Field(
        default_factory=list,
        description="Columns used for grouping."
    )

    order_by: str | None = Field(
        default=None,
        description="SQL ordering expression using allowed columns."
    )

    limit: int | None = Field(
        default=None,
        description="Maximum number of rows to return."
    )

    missing_slots: list[str] = Field(
        default_factory=list,
        description="Information still missing from the user request."
    )

    unsupported_terms: list[str] = Field(
        default_factory=list,
        description="User terms that cannot be mapped to the provided schema."
    )

    ambiguous_terms: list[str] = Field(
        default_factory=list,
        description="User terms that are ambiguous and require clarification."
    )

    @field_validator("time_range", mode="before")
    @classmethod
    def force_time_range_to_none(cls, value: object) -> None:
        """
        Schema mapping must never produce time_range.

        Even if the LLM tries to return a time_range object, discard it before validation.
        The graph injects the deterministic parsed_time_range later.
        """

        return None


class MessageClassificationOutput(BaseModel):
    message_type: UserMessageTypeOutput = Field(
        description="The classified type of the latest user message."
    )

    reason: str = Field(
        description="A short explanation of why this message type was selected."
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Classifier confidence between 0 and 1.",
    )

    active_request_action: ActiveRequestActionOutput = Field(
        description=(
            "How deterministic lifecycle code should relate the latest message to "
            "an active request. Use none only when no request mutation is needed."
        ),
    )

    correction_targets: list[CorrectionTargetOutput] = Field(
        default_factory=list,
        description=(
            "Canonical fields explicitly replaced by a correction: metric, location, "
            "time_range, aggregation, chart_type, output_type, group_by, order_by, "
            "limit, or filters."
        ),
    )

    resolved_request_type: Literal["data", "chart"] | None = Field(
        default=None,
        description=(
            "Request type for a new request, output-type correction, or ambiguous "
            "message that is later confirmed as a new request."
        ),
    )
