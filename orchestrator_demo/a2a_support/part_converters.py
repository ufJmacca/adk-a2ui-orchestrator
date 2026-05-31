"""Helpers for constructing and parsing local A2A transport parts."""

from typing import Any

from orchestrator_demo.a2a_support.transport import (
    A2UI_MIME_TYPE,
    A2uiUserAction,
    DataPart,
    TextPart,
)


def text_part_from_text(text: str) -> TextPart:
    return TextPart(text=text)


def a2ui_data_part_from_payload(payload: dict[str, Any]) -> DataPart:
    return DataPart(mimeType=A2UI_MIME_TYPE, data=payload)


def a2ui_user_action_from_part(part: DataPart) -> A2uiUserAction:
    return A2uiUserAction.model_validate(part.data)


__all__ = [
    "a2ui_data_part_from_payload",
    "a2ui_user_action_from_part",
    "text_part_from_text",
]
