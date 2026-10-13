from typing import Any
import ast


def extract_plain_text_from_message_like_value(value: Any) -> str:
    """
    Extract plain user text from LangGraph/LangChain-style message content.

    It supports:
    - plain strings
    - stringified content blocks, e.g. "[{'type': 'text', 'text': '...'}]"
    - dicts with text/content fields
    - lists of content blocks
    """

    if value is None:
        return ""

    if isinstance(value, str):
        stripped_value = value.strip()

        if not stripped_value:
            return ""

        extracted_lines: list[str] = []

        for line in stripped_value.splitlines():
            stripped_line = line.strip()

            if not stripped_line:
                continue

            try:
                parsed_line = ast.literal_eval(stripped_line)
            except (ValueError, SyntaxError):
                continue

            extracted_line = extract_plain_text_from_message_like_value(parsed_line)

            if extracted_line:
                extracted_lines.append(extracted_line)

        if extracted_lines:
            return "\n".join(extracted_lines)

        try:
            parsed_value = ast.literal_eval(stripped_value)
        except (ValueError, SyntaxError):
            return stripped_value

        extracted_value = extract_plain_text_from_message_like_value(parsed_value)

        return extracted_value or stripped_value

    if isinstance(value, dict):
        text_value = value.get("text")

        if isinstance(text_value, str):
            return text_value.strip()

        content_value = value.get("content")

        if content_value is not None:
            return extract_plain_text_from_message_like_value(content_value)

        return ""

    if isinstance(value, (list, tuple)):
        extracted_parts: list[str] = []

        for item in value:
            extracted_item = extract_plain_text_from_message_like_value(item)

            if extracted_item:
                extracted_parts.append(extracted_item)

        return "\n".join(extracted_parts)

    content_value = getattr(value, "content", None)

    if content_value is not None:
        return extract_plain_text_from_message_like_value(content_value)

    return str(value).strip()