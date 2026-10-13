from uuid import uuid4

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from agent.graph import build_graph


app = build_graph()


def create_run_config() -> RunnableConfig:
    thread_id = f"cli-chat-{uuid4()}"

    return {
        "configurable": {
            "thread_id": thread_id,
        }
    }


def print_last_assistant_message(result: dict) -> None:
    messages = result.get("messages", [])

    if not messages:
        return

    last_message = messages[-1]

    if last_message.type == "ai":
        print(f"\nAssistant: {last_message.content}\n")

def print_debug_state(result: dict) -> None:
    print("\n--- DEBUG STATE ---")
    print("user_message_type:", result.get("user_message_type"))
    print("classification_reason:", result.get("classification_reason"))
    print("request_status:", result.get("request_status"))
    print("active_request_type:", result.get("active_request_type"))
    print("active_request_text:", result.get("active_request_text"))
    print("clarification_answers:", result.get("clarification_answers"))
    print("mapping_input_text:", result.get("mapping_input_text"))
    print("metric_column:", result.get("metric_column"))
    print("time_range:", result.get("time_range"))
    print("filters:", result.get("filters"))
    print("missing_slots:", result.get("missing_slots"))
    print("is_complete:", result.get("is_complete"))
    print("error:", result.get("error"))
    print("-------------------\n")


def print_generated_sql_if_available(result: dict) -> None:
    if not result.get("sql_generated_in_current_turn"):
        return

    sql_query = result.get("sql_query")
    sql_is_safe = result.get("sql_is_safe")
    sql_validation_errors = result.get("sql_validation_errors", [])

    if not sql_query:
        return

    print("\nGenerated SQL:")
    print(sql_query)

    print("\nSQL safety:")
    print(sql_is_safe)

    if sql_validation_errors:
        print("\nSQL validation errors:")
        print(sql_validation_errors)

    print()


def run_chat() -> None:
    config = create_run_config()

    print("SQL Agent CLI")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        result = app.invoke(
            {
                "messages": [
                    HumanMessage(content=user_input)
                ]
            },
            config=config,
        )

        print_last_assistant_message(result)
        print_debug_state(result)

        if result.get("conversation_status") == "ended":
            break

        if result.get("request_status") == "sql_generated":
            print_generated_sql_if_available(result)


if __name__ == "__main__":
    run_chat()