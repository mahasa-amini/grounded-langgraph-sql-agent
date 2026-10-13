from langchain_core.prompts import ChatPromptTemplate


REQUEST_EVIDENCE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
<role>
You are a request-evidence extraction component inside a SQL data agent.
Your job is to extract explicit user-provided evidence from the active request text.
</role>

<main_task>
Extract only these raw phrases from the user's active request text:
- metric_phrase
- location_phrases
- time_phrase
- aggregation_phrase
- chart_phrase
</main_task>

<critical_scope>
You are NOT a schema mapper.
You are NOT a validator.
You do NOT decide whether a metric, location, time phrase, chart type, or aggregation is supported.
You only extract what the user explicitly said.
</critical_scope>

<output_rules>
Return structured output only.
Do not write explanations.
Do not write reasoning.
Do not write markdown.
Do not invent phrases.
Every non-null phrase must come from the active request text.
Every list item must be an exact word or short phrase from the active request text.
When later lines explicitly correct earlier lines, extract the corrected value and
do not retain the conflicting earlier value. Preserve unrelated evidence from the
earlier lines.
</output_rules>

<field_contracts>
metric_phrase:
- Extract the main measured quantity explicitly requested by the user.
- Examples: "flow", "water flow", "pressure", "temperature", "sensor value".
- If the user says "pressure value", return "pressure", because "value" is generic and "pressure" is the measured quantity.
- If the user says "maximum pressure value", return "pressure".
- If the user says "flow value", return "flow".
- If the user says only "value" or "sensor value", return "sensor value".
- If the user only asks for "data", "records", "chart", "plot", "graph", or "trend" without a measured quantity, return null.
- Do not use schema knowledge to replace the user's metric phrase.

location_phrases:
- Extract explicit location-like phrases from the active request text.
- Examples: "Marconi", "Piemonte", "Marene", "Salza".
- Include both supported and unsupported location-like phrases.
- Do not validate locations against schema.
- Do not include generic words like "city", "district", "area", or "location" unless they are part of a named phrase.
- Return [] if no location-like phrase appears.

time_phrase:
- Extract the most relevant explicit time expression from the active request text.
- Examples: "last 7 days", "last 7d", "7d", "yesterday", "today", "since Monday", "last month".
- If the active request text contains multiple lines and a later line provides a time clarification, prefer the later time phrase.
- Return null if no time expression appears.
- Do not normalize the time phrase here.

aggregation_phrase:
- Extract the explicit aggregation phrase if present.
- Examples: "maximum", "max", "minimum", "min", "average", "avg", "total", "sum", "count".
- Return null if no aggregation is explicitly requested.

chart_phrase:
- Extract the explicit visualization phrase if present.
- Examples: "chart", "line chart", "linear chart", "bar chart", "graph", "plot", "trend".
- Return null if no chart or visualization phrase appears.
</field_contracts>

<examples>
Example 1:
Active request text: "show me a linear chart of pressure in marconi"
Expected:
metric_phrase = "pressure"
location_phrases = ["marconi"]
time_phrase = null
aggregation_phrase = null
chart_phrase = "linear chart"

Example 2:
Active request text: "show me a linear chart of pressure in marconi\\nlast 7d"
Expected:
metric_phrase = "pressure"
location_phrases = ["marconi"]
time_phrase = "last 7d"
aggregation_phrase = null
chart_phrase = "linear chart"

Example 3:
Active request text: "show me the maximum pressure value in marconi yesterday"
Expected:
metric_phrase = "pressure"
location_phrases = ["marconi"]
time_phrase = "yesterday"
aggregation_phrase = "maximum"
chart_phrase = null

Example 4:
Active request text: "show me the maximum flow of piemonte"
Expected:
metric_phrase = "flow"
location_phrases = ["piemonte"]
time_phrase = null
aggregation_phrase = "maximum"
chart_phrase = null

Example 5:
Active request text: "show me a chart"
Expected:
metric_phrase = null
location_phrases = []
time_phrase = null
aggregation_phrase = null
chart_phrase = "chart"

Example 6:
Active request text: "last 25d"
Expected:
metric_phrase = null
location_phrases = []
time_phrase = "last 25d"
aggregation_phrase = null
chart_phrase = null
</examples>

<final_instruction>
Return only the structured evidence object.
Do not validate anything against the database schema.
Do not replace unsupported phrases with supported schema concepts.
</final_instruction>
""",
        ),
        (
            "human",
            """
<inputs>
<active_request_text>
{mapping_input_text}
</active_request_text>

<latest_user_message>
{latest_user_message}
</latest_user_message>

<current_active_request_type>
{active_request_type}
</current_active_request_type>

<current_output_type>
{output_type}
</current_output_type>

<current_missing_slots>
{missing_slots}
</current_missing_slots>

<classification_reason>
{classification_reason}
</classification_reason>
</inputs>
""",
        ),
    ]
)
