"""Specialist descriptor validation for the dynamic agent registry."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import ValidationError

from orchestrator_demo.contracts import AgentDescriptor


REQUIRED_SPECIALIST_AGENT_IDS = frozenset(
    {
        "industry_research",
        "web_search",
        "internal_knowledge",
        "credit_risk",
        "relationship_summary",
        "product_opportunity",
        "compliance_policy",
        "data_quality",
        "meeting_prep",
        "synthesis",
    }
)

JSON_SCHEMA_TYPES = frozenset(
    {
        "array",
        "boolean",
        "integer",
        "null",
        "number",
        "object",
        "string",
    }
)
SECRET_FIELD_MARKERS = (
    "api_key",
    "apikey",
    "access_key",
    "secret",
    "password",
    "token",
    "credential",
    "private_key",
    "openrouter_api_key",
)


class DescriptorValidationError(ValueError):
    """Raised when an agent descriptor config cannot be safely loaded."""


def validate_agent_descriptors(raw_descriptors: Any) -> dict[str, AgentDescriptor]:
    """Validate config-sourced descriptors and return them keyed by agent id."""

    if not isinstance(raw_descriptors, Sequence) or isinstance(
        raw_descriptors, (str, bytes, bytearray)
    ):
        raise DescriptorValidationError("AVAILABLE_AGENTS must be a sequence")

    descriptors_by_id: dict[str, AgentDescriptor] = {}
    for index, raw_descriptor in enumerate(raw_descriptors):
        location = f"AVAILABLE_AGENTS[{index}]"
        _reject_secret_like_fields(raw_descriptor, location)
        descriptor = _coerce_descriptor(raw_descriptor, location)
        _validate_descriptor_shape(descriptor, location)

        if descriptor.agent_id in descriptors_by_id:
            raise DescriptorValidationError(
                f"duplicate agent_id in registry config: {descriptor.agent_id}"
            )
        descriptors_by_id[descriptor.agent_id] = descriptor

    return descriptors_by_id


def _coerce_descriptor(raw_descriptor: Any, location: str) -> AgentDescriptor:
    if isinstance(raw_descriptor, AgentDescriptor):
        return raw_descriptor

    try:
        return AgentDescriptor.model_validate(raw_descriptor)
    except ValidationError as exc:
        raise DescriptorValidationError(
            f"invalid descriptor at {location}: {_redacted_validation_message(exc)}"
        ) from exc


def _validate_descriptor_shape(
    descriptor: AgentDescriptor,
    location: str,
) -> None:
    list_fields = {
        "capabilities": descriptor.capabilities,
        "a2ui_catalogs": descriptor.a2ui_catalogs,
        "routing_examples": descriptor.routing_examples,
    }
    for field_name, values in list_fields.items():
        if not values or any(not isinstance(value, str) or not value.strip() for value in values):
            raise DescriptorValidationError(
                f"{location}.{field_name} must contain at least one non-empty string"
            )

    _validate_json_schema(descriptor.input_schema, f"{location}.input_schema")
    _validate_json_schema(descriptor.output_schema, f"{location}.output_schema")


def _validate_json_schema(schema: Any, location: str) -> None:
    if not isinstance(schema, Mapping):
        raise DescriptorValidationError(f"{location} must be a JSON-schema object")

    schema_type = schema.get("type")
    if schema_type is None:
        raise DescriptorValidationError(f"{location} must declare a JSON-schema type")

    invalid_types: list[str] = []
    if isinstance(schema_type, str):
        if schema_type not in JSON_SCHEMA_TYPES:
            invalid_types.append(schema_type)
    elif isinstance(schema_type, Sequence) and not isinstance(
        schema_type, (bytes, bytearray)
    ):
        for type_name in schema_type:
            if not isinstance(type_name, str) or type_name not in JSON_SCHEMA_TYPES:
                invalid_types.append(str(type_name))
    else:
        invalid_types.append(str(schema_type))

    if invalid_types:
        invalid = ", ".join(invalid_types)
        raise DescriptorValidationError(
            f"{location} has invalid JSON-schema type: {invalid}"
        )

    properties = schema.get("properties")
    if properties is not None:
        if not isinstance(properties, Mapping):
            raise DescriptorValidationError(f"{location}.properties must be an object")
        for property_name, property_schema in properties.items():
            _validate_json_schema(
                property_schema,
                f"{location}.properties.{property_name}",
            )

    items = schema.get("items")
    if items is not None:
        _validate_json_schema(items, f"{location}.items")

    required = schema.get("required")
    if required is not None and (
        not isinstance(required, Sequence)
        or isinstance(required, (str, bytes, bytearray))
        or any(not isinstance(field_name, str) for field_name in required)
    ):
        raise DescriptorValidationError(f"{location}.required must be a string list")


def _reject_secret_like_fields(value: Any, path: str) -> None:
    if isinstance(value, AgentDescriptor):
        value = value.model_dump()

    if isinstance(value, Mapping):
        for key, child_value in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}"
            if _is_secret_like_field_name(key_text):
                raise DescriptorValidationError(
                    f"descriptor config contains secret-like field: {child_path}"
                )
            _reject_secret_like_fields(child_value, child_path)
        return

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child_value in enumerate(value):
            _reject_secret_like_fields(child_value, f"{path}[{index}]")


def _is_secret_like_field_name(field_name: str) -> bool:
    normalized = field_name.lower().replace("-", "_")
    return any(marker in normalized for marker in SECRET_FIELD_MARKERS)


def _redacted_validation_message(exc: ValidationError) -> str:
    messages: list[str] = []
    for error in exc.errors(include_url=False, include_input=False):
        location = ".".join(str(part) for part in error["loc"])
        messages.append(f"{location}: {error['msg']}")
    return "; ".join(messages)


__all__ = [
    "DescriptorValidationError",
    "REQUIRED_SPECIALIST_AGENT_IDS",
    "validate_agent_descriptors",
]
