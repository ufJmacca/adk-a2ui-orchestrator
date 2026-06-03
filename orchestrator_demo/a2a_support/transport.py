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


def _dump_sdk_model(value: Any) -> Any:
    root = getattr(value, "root", None)
    if root is not None:
        value = root

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(by_alias=True, mode="json", exclude_none=True)

    return value


A2APartPayload = Annotated[
    Annotated[TextPart, Tag("text")] | Annotated[DataPart, Tag("data")],
    Discriminator(_part_discriminator),
]


class A2AArtifact(TransportModel):
    artifact_id: str = Field(min_length=1, alias="artifactId")
    description: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    extensions: list[str] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        exclude_if=lambda value: not value,
    )
    name: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    parts: list[A2APartPayload] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def normalize_sdk_artifact_model(cls, value: Any) -> Any:
        value = _dump_sdk_model(value)
        if not isinstance(value, dict):
            return value

        normalized = dict(value)
        if normalized.get("metadata") is None:
            normalized.pop("metadata", None)
        return normalized


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

    @model_validator(mode="before")
    @classmethod
    def normalize_sdk_message_wire_fields(cls, value: Any) -> Any:
        return _normalize_sdk_message_fields(
            value,
            fallback_timestamp=None,
            task_id=None,
            context_id=None,
        )


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
        if not isinstance(value, dict):
            model_dump = getattr(value, "model_dump", None)
            if callable(model_dump):
                value = model_dump(by_alias=True, mode="json")

        if not isinstance(value, dict):
            return value

        normalized = dict(value)
        task_id = _first_present(normalized, "taskId", "task_id")
        context_id = _first_present(normalized, "contextId", "context_id")
        message = normalized.get("message")
        if message is not None:
            normalized["message"] = _normalize_sdk_message_fields(
                message,
                fallback_timestamp=normalized.get("timestamp"),
                task_id=task_id,
                context_id=context_id,
            )

        return normalized

    @model_validator(mode="after")
    def validate_child_message_shares_task_context(self) -> "TaskStatusUpdate":
        if self.message is None:
            return self

        if self.message.task_id != self.task_id:
            raise ValueError("status message taskId must match status taskId")
        if self.message.context_id != self.context_id:
            raise ValueError("status message contextId must match status contextId")

        return self


def _first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value

    return None


def _coalesce_alias_pair(
    mapping: dict[str, Any],
    *,
    alias_key: str,
    field_name: str,
) -> None:
    if alias_key not in mapping or field_name not in mapping:
        return

    alias_value = mapping[alias_key]
    field_value = mapping[field_name]
    if alias_value is not None and field_value is not None and alias_value != field_value:
        raise ValueError(f"{alias_key} and {field_name} must match")
    if alias_value is None and field_value is not None:
        mapping[alias_key] = field_value
    mapping.pop(field_name, None)


def _setdefault_child_id(
    mapping: dict[str, Any],
    *,
    alias_key: str,
    field_name: str,
    value: Any,
) -> None:
    _coalesce_alias_pair(mapping, alias_key=alias_key, field_name=field_name)
    if (
        value is not None
        and mapping.get(alias_key) is None
        and mapping.get(field_name) is None
    ):
        mapping[alias_key] = value
        mapping.pop(field_name, None)


def _pop_empty_unsupported_sdk_field(
    mapping: dict[str, Any],
    *,
    field_label: str,
    keys: tuple[str, ...],
) -> None:
    for key in keys:
        if key not in mapping:
            continue

        value = mapping.pop(key)
        if value is None:
            continue
        if isinstance(value, (list, tuple, set, dict)) and len(value) == 0:
            continue

        raise ValueError(
            f"{field_label} are not supported by the local A2A transport"
        )


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
    artifacts: list[A2AArtifact] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
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
        kind = normalized.pop("kind", None)
        if kind is not None and kind != "task":
            raise ValueError("kind must be task")
        if normalized.get("artifacts") is None:
            normalized.pop("artifacts", None)
        if normalized.get("history") is None:
            normalized.pop("history", None)
        if normalized.get("messages") is None:
            normalized.pop("messages", None)

        status = normalized.get("status")
        if status is not None and not isinstance(status, dict):
            model_dump = getattr(status, "model_dump", None)
            if callable(model_dump):
                status = model_dump(by_alias=True, mode="json")

        task_id = _first_present(normalized, "id", "taskId", "task_id")
        context_id = _first_present(normalized, "contextId", "context_id")
        if isinstance(status, dict):
            normalized_status = dict(status)
            _setdefault_child_id(
                normalized_status,
                alias_key="taskId",
                field_name="task_id",
                value=task_id,
            )
            _setdefault_child_id(
                normalized_status,
                alias_key="contextId",
                field_name="context_id",
                value=context_id,
            )
            status_timestamp = normalized_status.get("timestamp")
            message = normalized_status.get("message")
            if message is not None:
                normalized_status["message"] = _normalize_sdk_message_fields(
                    message,
                    fallback_timestamp=status_timestamp,
                    task_id=task_id,
                    context_id=context_id,
                )
            normalized["status"] = normalized_status

        status_timestamp = None
        if isinstance(normalized.get("status"), dict):
            status_timestamp = normalized["status"].get("timestamp")
        messages = normalized.get("history", normalized.get("messages"))
        if isinstance(messages, list):
            normalized["history"] = [
                _normalize_sdk_message_fields(
                    message,
                    fallback_timestamp=status_timestamp,
                    task_id=task_id,
                    context_id=context_id,
                )
                for message in messages
            ]
            normalized.pop("messages", None)

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
        if value.metadata:
            wire_status["metadata"] = value.metadata

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


def _normalize_sdk_message_fields(
    value: Any,
    *,
    fallback_timestamp: Any,
    task_id: Any,
    context_id: Any,
) -> Any:
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
    kind = normalized.pop("kind", None)
    if kind is not None and kind != "message":
        raise ValueError("message kind must be message")
    _pop_empty_unsupported_sdk_field(
        normalized,
        field_label="message extensions",
        keys=("extensions",),
    )
    _pop_empty_unsupported_sdk_field(
        normalized,
        field_label="message referenceTaskIds",
        keys=("referenceTaskIds", "reference_task_ids"),
    )
    if normalized.get("timestamp") is None and fallback_timestamp is not None:
        normalized["timestamp"] = fallback_timestamp
    _coalesce_alias_pair(
        normalized,
        alias_key="messageId",
        field_name="message_id",
    )
    _setdefault_child_id(
        normalized,
        alias_key="taskId",
        field_name="task_id",
        value=task_id,
    )
    _setdefault_child_id(
        normalized,
        alias_key="contextId",
        field_name="context_id",
        value=context_id,
    )

    return normalized


class A2uiUserAction(UserAction):
    """A2UI userAction event parsed from renderer or DataPart payloads."""


__all__ = [
    "A2UI_MIME_TYPE",
    "A2AArtifact",
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
