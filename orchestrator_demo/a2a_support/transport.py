"""Local A2A/A2UI transport envelope contracts.

These models define the JSON shape used by the local harness before the demo
is wired to a full A2A runtime.
"""

from datetime import UTC, datetime
from typing import Annotated, Any, Final, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    Tag,
    field_serializer,
    field_validator,
)
from pydantic import model_validator

from orchestrator_demo.contracts import UserAction


A2UI_MIME_TYPE: Final[Literal["application/json+a2ui"]] = "application/json+a2ui"

MessageRole = Literal["user", "agent"]
TaskState = Literal[
    "submitted",
    "working",
    "input-required",
    "completed",
    "canceled",
    "failed",
    "rejected",
    "auth-required",
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

    @field_validator("metadata", check_fields=False, mode="before")
    @classmethod
    def normalize_null_metadata(cls, value: Any) -> Any:
        if value is None:
            return {}

        return value


class A2APart(TransportModel):
    type: Literal["text", "data"]

    @model_validator(mode="before")
    @classmethod
    def normalize_sdk_kind_discriminator(cls, value: Any) -> Any:
        root = getattr(value, "root", None)
        if root is not None:
            value = root

        if not isinstance(value, dict):
            model_dump = getattr(value, "model_dump", None)
            if callable(model_dump):
                value = model_dump(by_alias=True, mode="json")

        if not isinstance(value, dict):
            return value

        normalized = dict(value)
        if normalized.get("metadata") is None:
            normalized.pop("metadata", None)

        kind = normalized.pop("kind", None)
        if kind is None:
            return normalized

        local_type = normalized.get("type")
        if local_type is not None and local_type != kind:
            raise ValueError("kind and type must match")

        normalized["type"] = kind
        return normalized


class TextPart(A2APart):
    type: Literal["text"] = "text"
    text: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        exclude_if=lambda value: not value,
    )


class DataPart(A2APart):
    type: Literal["data"] = "data"
    data: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def normalize_a2ui_mime_type(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        normalized = dict(value)
        metadata = normalized.get("metadata")
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            return normalized

        metadata = dict(metadata)
        wire_mime_type = normalized.pop("mimeType", None)
        python_mime_type = normalized.pop("mime_type", None)
        if (
            wire_mime_type is not None
            and python_mime_type is not None
            and wire_mime_type != python_mime_type
        ):
            raise ValueError("mimeType and mime_type must match")

        mime_type = wire_mime_type if wire_mime_type is not None else python_mime_type
        if mime_type is not None:
            existing_mime_type = metadata.get("mimeType")
            if existing_mime_type is not None and existing_mime_type != mime_type:
                raise ValueError("metadata.mimeType must match mimeType")
            metadata["mimeType"] = mime_type

        normalized["metadata"] = metadata
        return normalized

    @model_validator(mode="after")
    def validate_a2ui_mime_type(self) -> "DataPart":
        if self.metadata.get("mimeType") != A2UI_MIME_TYPE:
            raise ValueError("metadata.mimeType must be application/json+a2ui")

        return self

    @property
    def mime_type(self) -> Literal["application/json+a2ui"]:
        return A2UI_MIME_TYPE


def _part_discriminator(value: Any) -> str | None:
    if isinstance(value, dict):
        local_type = value.get("type")
        if local_type is not None:
            return str(local_type)
        kind = value.get("kind")
        if kind is not None:
            return str(kind)
        return None

    root = getattr(value, "root", None)
    if root is not None:
        value = root

    part_type = getattr(value, "type", None)
    if part_type is None:
        part_type = getattr(value, "kind", None)
    if part_type is None:
        return None
    return str(part_type)


A2APartPayload = Annotated[
    Annotated[TextPart, Tag("text")] | Annotated[DataPart, Tag("data")],
    Discriminator(_part_discriminator),
]


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
    status: TaskState = Field(
        validation_alias=AliasChoices("state", "status"),
        serialization_alias="state",
    )
    timestamp: datetime
    message: A2AMessage | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def normalize_sdk_status_model(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return value

        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            return model_dump(by_alias=True, mode="json")

        return value


class A2ATask(TransportModel):
    task_id: str = Field(
        pattern=r"^task_[A-Za-z0-9][A-Za-z0-9_-]*$",
        validation_alias=AliasChoices("id", "taskId"),
        serialization_alias="id",
    )
    context_id: str = Field(
        pattern=r"^ctx_[A-Za-z0-9][A-Za-z0-9_-]*$",
        alias="contextId",
    )
    status: TaskStatusUpdate
    messages: list[A2AMessage] = Field(
        default_factory=list,
        validation_alias=AliasChoices("history", "messages"),
        serialization_alias="history",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def normalize_task_wire_fields(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            model_dump = getattr(value, "model_dump", None)
            if callable(model_dump):
                value = model_dump(by_alias=True, mode="json")

        if not isinstance(value, dict):
            return value

        normalized = dict(value)
        status = normalized.get("status")
        if status is not None and not isinstance(status, dict):
            model_dump = getattr(status, "model_dump", None)
            if callable(model_dump):
                status = model_dump(by_alias=True, mode="json")

        if isinstance(status, dict):
            normalized_status = dict(status)
            task_id = normalized.get("id") or normalized.get("taskId")
            context_id = normalized.get("contextId")
            if task_id is not None:
                normalized_status.setdefault("taskId", task_id)
            if context_id is not None:
                normalized_status.setdefault("contextId", context_id)
            normalized["status"] = normalized_status

        return normalized

    @field_serializer("status", when_used="json")
    def serialize_a2a_task_status(
        self,
        value: TaskStatusUpdate,
    ) -> dict[str, Any]:
        wire_status: dict[str, Any] = {
            "state": value.status,
            "timestamp": value.serialize_timestamp(value.timestamp),
        }
        if value.message is not None:
            wire_status["message"] = value.message.model_dump(
                by_alias=True,
                mode="json",
            )

        return wire_status

    @model_validator(mode="after")
    def validate_child_envelopes_share_task_context(self) -> "A2ATask":
        if self.status.task_id != self.task_id:
            raise ValueError("status taskId must match task taskId")
        if self.status.context_id != self.context_id:
            raise ValueError("status contextId must match task contextId")
        if self.status.message is not None:
            if self.status.message.task_id != self.task_id:
                raise ValueError("status message taskId must match task taskId")
            if self.status.message.context_id != self.context_id:
                raise ValueError("status message contextId must match task contextId")

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
