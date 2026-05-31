"""Local A2A/A2UI transport envelope contracts.

These models define the JSON shape used by the local harness before the demo
is wired to a full A2A runtime.
"""

from datetime import UTC, datetime
from typing import Annotated, Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator
from pydantic import model_validator

from orchestrator_demo.contracts import UserAction


A2UI_MIME_TYPE: Final[Literal["application/json+a2ui"]] = "application/json+a2ui"

MessageRole = Literal["user", "agent"]
TaskState = Literal[
    "submitted",
    "working",
    "input_required",
    "completed",
    "canceled",
    "failed",
    "rejected",
    "auth_required",
    "unknown",
]


class TransportModel(BaseModel):
    """Strict base model for local transport envelopes."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @field_validator("timestamp", check_fields=False)
    @classmethod
    def require_timezone_aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must include timezone information")

        return value

    @field_serializer("timestamp", check_fields=False, when_used="json")
    def serialize_timestamp(self, value: datetime) -> str:
        utc_value = value.astimezone(UTC)
        timestamp = utc_value.replace(tzinfo=None).isoformat(timespec="seconds")
        return f"{timestamp}Z"


class A2APart(TransportModel):
    type: Literal["text", "data"]


class TextPart(A2APart):
    type: Literal["text"] = "text"
    text: str = Field(min_length=1)


class DataPart(A2APart):
    type: Literal["data"] = "data"
    mime_type: Literal["application/json+a2ui"] = Field(alias="mimeType")
    data: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)


A2APartPayload = Annotated[TextPart | DataPart, Field(discriminator="type")]


class A2AMessage(TransportModel):
    message_id: str = Field(
        pattern=r"^msg_[A-Za-z0-9][A-Za-z0-9_-]*$",
        alias="messageId",
    )
    context_id: str = Field(
        pattern=r"^ctx_[A-Za-z0-9][A-Za-z0-9_-]*$",
        alias="contextId",
    )
    task_id: str = Field(
        pattern=r"^task_[A-Za-z0-9][A-Za-z0-9_-]*$",
        alias="taskId",
    )
    role: MessageRole
    timestamp: datetime
    parts: list[A2APartPayload] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskStatusUpdate(TransportModel):
    task_id: str = Field(
        pattern=r"^task_[A-Za-z0-9][A-Za-z0-9_-]*$",
        alias="taskId",
    )
    context_id: str = Field(
        pattern=r"^ctx_[A-Za-z0-9][A-Za-z0-9_-]*$",
        alias="contextId",
    )
    status: TaskState
    timestamp: datetime
    message: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class A2ATask(TransportModel):
    task_id: str = Field(
        pattern=r"^task_[A-Za-z0-9][A-Za-z0-9_-]*$",
        alias="taskId",
    )
    context_id: str = Field(
        pattern=r"^ctx_[A-Za-z0-9][A-Za-z0-9_-]*$",
        alias="contextId",
    )
    status: TaskStatusUpdate
    messages: list[A2AMessage] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_child_envelopes_share_task_context(self) -> "A2ATask":
        if self.status.task_id != self.task_id:
            raise ValueError("status taskId must match task taskId")
        if self.status.context_id != self.context_id:
            raise ValueError("status contextId must match task contextId")

        for message in self.messages:
            if message.task_id != self.task_id:
                raise ValueError("message taskId must match task taskId")
            if message.context_id != self.context_id:
                raise ValueError("message contextId must match task contextId")

        return self


class A2uiUserAction(UserAction):
    """A2UI userAction event parsed from renderer or DataPart payloads."""


__all__ = [
    "A2UI_MIME_TYPE",
    "A2AMessage",
    "A2APart",
    "A2APartPayload",
    "A2ATask",
    "A2uiUserAction",
    "DataPart",
    "MessageRole",
    "TaskState",
    "TaskStatusUpdate",
    "TextPart",
    "TransportModel",
]
