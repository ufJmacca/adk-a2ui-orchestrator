"""Helpers for constructing and parsing local A2A transport parts."""

from typing import Any, overload

from pydantic import ValidationError

from orchestrator_demo.a2a_support.transport import (
    A2UI_MIME_TYPE,
    A2uiUserAction,
    DataPart,
    TextPart,
)
from orchestrator_demo.a2ui_support.event_parser import (
    PlanUserActionParseError,
    StructuredUserActionRequiredError,
    parse_user_action,
)
from orchestrator_demo.contracts import A2uiPayload


def text_part_from_text(text: str) -> TextPart:
    return TextPart(text=text)


@overload
def a2ui_data_part_from_payload(payload: dict[str, Any]) -> DataPart: ...


@overload
def a2ui_data_part_from_payload(payload: list[dict[str, Any]]) -> list[DataPart]: ...


def a2ui_data_part_from_payload(payload: A2uiPayload) -> DataPart | list[DataPart]:
    if isinstance(payload, list):
        return a2ui_data_parts_from_payload(payload)

    return DataPart(data=payload, metadata={"mimeType": A2UI_MIME_TYPE})


def a2ui_data_parts_from_payload(payload: A2uiPayload) -> list[DataPart]:
    if isinstance(payload, list):
        return [
            DataPart(data=message, metadata={"mimeType": A2UI_MIME_TYPE})
            for message in payload
        ]

    return [DataPart(data=payload, metadata={"mimeType": A2UI_MIME_TYPE})]


def a2ui_user_action_from_part(part: DataPart) -> A2uiUserAction:
    if not isinstance(part.data, dict):
        raise ValueError("A2UI userAction DataPart data must be an object")

    try:
        action = parse_user_action(part)
        return A2uiUserAction.model_validate(
            action.model_dump(by_alias=True, mode="json")
        )
    except (
        PlanUserActionParseError,
        StructuredUserActionRequiredError,
        ValidationError,
    ):
        raise ValueError("invalid A2UI userAction DataPart payload") from None


__all__ = [
    "a2ui_data_part_from_payload",
    "a2ui_data_parts_from_payload",
    "a2ui_user_action_from_part",
    "text_part_from_text",
]
