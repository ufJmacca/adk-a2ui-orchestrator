"""Specialist descriptor validation for the dynamic agent registry."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
import re
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
JSON_SCHEMA_LIST_CONTAINERS = ("allOf", "anyOf", "oneOf", "prefixItems")
JSON_SCHEMA_MAP_CONTAINERS = (
    "$defs",
    "definitions",
    "dependentSchemas",
    "patternProperties",
)
JSON_SCHEMA_REDACTED_KEY_CONTAINERS = JSON_SCHEMA_MAP_CONTAINERS + (
    "dependentRequired",
)
JSON_SCHEMA_VALUE_CONTAINERS = (
    "not",
    "if",
    "then",
    "else",
    "contains",
    "propertyNames",
    "unevaluatedItems",
    "unevaluatedProperties",
)
SECRET_FIELD_MARKERS = (
    "api_key",
    "apikey",
    "access_key",
    "secret",
    "password",
    "token",
    "credential",
    "authorization",
    "private_key",
    "openrouter_api_key",
)
SECRET_VALUE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----",
        r"(?<![A-Za-z0-9])sk-[A-Za-z0-9][A-Za-z0-9_-]{8,}",
        r"(?<![A-Za-z0-9])(?:ghp|gho|ghu|ghs|github_pat)_[A-Za-z0-9_]{20,}",
        r"(?<![A-Za-z0-9])xox[baprs]-[A-Za-z0-9-]{10,}",
        r"(?<![A-Za-z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}",
        r"(?<![A-Za-z0-9])AIza[0-9A-Za-z_-]{20,}",
        r"(?<![A-Za-z0-9])eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}",
        r"(?<![A-Za-z0-9])bearer\s+[A-Za-z0-9._~+/=-]{6,}",
        r"(?<![A-Za-z0-9])authorization\b\s*[:=]\s*bearer\s+\S{6,}",
        r"(?<![A-Za-z0-9])(?:api[_-]?key|access[_-]?key|private[_-]?key|secret|password|token|credential)\b\s*[:=]\s*\S{6,}",
    )
)
REDACTED_SCHEMA_PATH_SEGMENT = "<redacted>"


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
        _reject_secret_like_fields(descriptor, location)
        _validate_descriptor_shape(descriptor, location)

        if descriptor.agent_id in descriptors_by_id:
            raise DescriptorValidationError(
                f"duplicate agent_id in registry config: {descriptor.agent_id}"
            )
        descriptors_by_id[descriptor.agent_id] = descriptor

    return descriptors_by_id


def _coerce_descriptor(raw_descriptor: Any, location: str) -> AgentDescriptor:
    if isinstance(raw_descriptor, AgentDescriptor):
        raw_descriptor = raw_descriptor.model_dump(warnings=False)

    try:
        return AgentDescriptor.model_validate(raw_descriptor)
    except ValidationError as exc:
        raise DescriptorValidationError(
            f"invalid descriptor at {location}: {_redacted_validation_message(exc)}"
        ) from None


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


def _validate_json_schema(
    schema: Any,
    location: str,
    *,
    allow_boolean: bool = False,
) -> None:
    if isinstance(schema, bool) and allow_boolean:
        return

    if not isinstance(schema, Mapping):
        raise DescriptorValidationError(f"{location} must be a JSON-schema object")

    _validate_required_fields(schema, location)

    if "type" in schema:
        _validate_json_schema_type(schema["type"], location)

    if "$ref" in schema and not isinstance(schema["$ref"], str):
        raise DescriptorValidationError(f"{location}.$ref must be a string")

    properties = schema.get("properties")
    if properties is not None:
        if not isinstance(properties, Mapping):
            raise DescriptorValidationError(f"{location}.properties must be an object")
        properties_location = f"{location}.properties"
        for property_name, property_schema in properties.items():
            safe_property_name = _validated_mapping_key_path_segment(
                properties_location,
                property_name,
            )
            _validate_json_schema(
                property_schema,
                f"{properties_location}.{safe_property_name}",
                allow_boolean=True,
            )

    items = schema.get("items")
    if items is not None:
        _validate_json_schema(items, f"{location}.items", allow_boolean=True)

    for container_name in JSON_SCHEMA_VALUE_CONTAINERS:
        _validate_schema_value_container(schema, container_name, location)

    for container_name in JSON_SCHEMA_LIST_CONTAINERS:
        _validate_schema_list_container(schema, container_name, location)

    for container_name in JSON_SCHEMA_MAP_CONTAINERS:
        _validate_schema_map_container(schema, container_name, location)

    additional_properties = schema.get("additionalProperties")
    if isinstance(additional_properties, Mapping):
        _validate_json_schema(
            additional_properties,
            f"{location}.additionalProperties",
            allow_boolean=True,
        )
    elif additional_properties is not None and not isinstance(
        additional_properties,
        bool,
    ):
        raise DescriptorValidationError(
            f"{location}.additionalProperties must be a JSON-schema object or boolean"
        )


def _validate_required_fields(schema: Mapping[str, Any], location: str) -> None:
    required = schema.get("required")
    if required is not None:
        if (
            not isinstance(required, Sequence)
            or isinstance(required, (str, bytes, bytearray))
            or any(not isinstance(field_name, str) for field_name in required)
        ):
            raise DescriptorValidationError(f"{location}.required must be a string list")

        for index, field_name in enumerate(required):
            if _is_secret_like_field_name(field_name):
                raise DescriptorValidationError(
                    f"descriptor config contains secret-like field: "
                    f"{location}.required[{index}]"
                )

    dependent_required = schema.get("dependentRequired")
    if dependent_required is not None:
        if not isinstance(dependent_required, Mapping):
            raise DescriptorValidationError(
                f"{location}.dependentRequired must be an object"
            )

        for property_name, field_names in dependent_required.items():
            safe_property_name = _safe_path_component(property_name)
            if not isinstance(property_name, str):
                raise DescriptorValidationError(
                    f"{location}.dependentRequired must map strings to string lists"
                )
            if _is_secret_like_value(property_name):
                raise DescriptorValidationError(
                    "descriptor config contains secret-like schema map key: "
                    f"{location}.dependentRequired.{REDACTED_SCHEMA_PATH_SEGMENT}"
                )
            safe_property_name = _schema_map_path_segment(property_name)
            if _is_secret_like_field_name(property_name):
                raise DescriptorValidationError(
                    f"descriptor config contains secret-like field: "
                    f"{location}.dependentRequired.{safe_property_name}"
                )
            if (
                not isinstance(field_names, Sequence)
                or isinstance(field_names, (str, bytes, bytearray))
                or any(not isinstance(field_name, str) for field_name in field_names)
            ):
                raise DescriptorValidationError(
                    f"{location}.dependentRequired.{safe_property_name} "
                    "must be a string list"
                )

            for index, field_name in enumerate(field_names):
                if _is_secret_like_field_name(field_name):
                    raise DescriptorValidationError(
                        f"descriptor config contains secret-like field: "
                        f"{location}.dependentRequired.{safe_property_name}[{index}]"
                    )


def _validate_json_schema_type(schema_type: Any, location: str) -> None:
    invalid_types: list[str] = []
    if isinstance(schema_type, str):
        if schema_type not in JSON_SCHEMA_TYPES:
            invalid_types.append(schema_type)
    elif isinstance(schema_type, Sequence) and not isinstance(
        schema_type, (bytes, bytearray)
    ):
        if not schema_type:
            invalid_types.append("<empty>")
        else:
            for type_name in schema_type:
                if not isinstance(type_name, str) or type_name not in JSON_SCHEMA_TYPES:
                    invalid_type = (
                        type_name
                        if isinstance(type_name, str)
                        else type(type_name).__name__
                    )
                    invalid_types.append(invalid_type)
    else:
        invalid_types.append(type(schema_type).__name__)

    if not invalid_types:
        return

    message = f"{location}.type has invalid JSON-schema type"
    safe_invalid_types = [
        type_name
        for type_name in invalid_types
        if type_name != "<empty>" and not _is_secret_like_value(type_name)
    ]
    if safe_invalid_types and len(safe_invalid_types) == len(invalid_types):
        message = f"{message}: {', '.join(safe_invalid_types)}"

    raise DescriptorValidationError(message)


def _validate_schema_value_container(
    schema: Mapping[str, Any],
    container_name: str,
    location: str,
) -> None:
    if container_name not in schema:
        return

    nested_schema = schema[container_name]
    _validate_json_schema(
        nested_schema,
        f"{location}.{container_name}",
        allow_boolean=True,
    )


def _validate_schema_list_container(
    schema: Mapping[str, Any],
    container_name: str,
    location: str,
) -> None:
    nested_schemas = schema.get(container_name)
    if nested_schemas is None:
        return

    if not isinstance(nested_schemas, Sequence) or isinstance(
        nested_schemas,
        (str, bytes, bytearray),
    ):
        raise DescriptorValidationError(
            f"{location}.{container_name} must be a JSON-schema list"
        )

    for index, nested_schema in enumerate(nested_schemas):
        _validate_json_schema(
            nested_schema,
            f"{location}.{container_name}[{index}]",
            allow_boolean=True,
        )


def _validate_schema_map_container(
    schema: Mapping[str, Any],
    container_name: str,
    location: str,
) -> None:
    nested_schemas = schema.get(container_name)
    if nested_schemas is None:
        return

    if not isinstance(nested_schemas, Mapping):
        raise DescriptorValidationError(
            f"{location}.{container_name} must be a JSON-schema object"
        )

    nested_location = f"{location}.{container_name}"
    for nested_name, nested_schema in nested_schemas.items():
        safe_nested_name = _validated_mapping_key_path_segment(
            nested_location,
            nested_name,
        )
        _validate_json_schema(
            nested_schema,
            f"{nested_location}.{safe_nested_name}",
            allow_boolean=True,
        )


def _reject_secret_like_fields(
    value: Any,
    path: str,
    *,
    schema_scan_context: str = "normal",
) -> None:
    if isinstance(value, AgentDescriptor):
        value = value.model_dump(warnings=False)

    if isinstance(value, str):
        if schema_scan_context != "schema_type" and _is_secret_like_value(value):
            raise DescriptorValidationError(
                f"descriptor config contains secret-like value: {path}"
            )
        return

    if isinstance(value, (bytes, bytearray)):
        try:
            value_text = bytes(value).decode("utf-8")
        except UnicodeDecodeError:
            return
        if schema_scan_context != "schema_type" and _is_secret_like_value(value_text):
            raise DescriptorValidationError(
                f"descriptor config contains secret-like value: {path}"
            )
        return

    if isinstance(value, Mapping):
        for key, child_value in value.items():
            safe_key = _validated_mapping_key_path_segment(path, key)
            child_path = f"{path}.{safe_key}"
            key_text = str(key)
            if _is_secret_like_field_name(key_text):
                raise DescriptorValidationError(
                    f"descriptor config contains secret-like field: {child_path}"
                )
            _reject_secret_like_fields(
                child_value,
                child_path,
                schema_scan_context=_schema_secret_scan_child_context(
                    schema_scan_context,
                    key_text,
                    child_path,
                ),
            )
        return

    if isinstance(value, Collection):
        for index, child_value in enumerate(value):
            _reject_secret_like_fields(
                child_value,
                f"{path}[{index}]",
                schema_scan_context=_schema_secret_scan_collection_child_context(
                    schema_scan_context,
                ),
            )


def _schema_secret_scan_child_context(
    parent_context: str,
    key: str,
    child_path: str,
) -> str:
    if parent_context == "normal" and _is_descriptor_schema_root_path(child_path):
        return "json_schema"

    if parent_context in {"schema_properties", "schema_map"}:
        return "json_schema"

    if parent_context != "json_schema":
        return "normal"

    if key == "type":
        return "schema_type"
    if key == "properties":
        return "schema_properties"
    if key in JSON_SCHEMA_MAP_CONTAINERS:
        return "schema_map"
    if key in JSON_SCHEMA_LIST_CONTAINERS:
        return "schema_list"
    if (
        key == "items"
        or key == "additionalProperties"
        or key in JSON_SCHEMA_VALUE_CONTAINERS
    ):
        return "json_schema"

    return "normal"


def _schema_secret_scan_collection_child_context(parent_context: str) -> str:
    if parent_context == "schema_type":
        return "schema_type"
    if parent_context == "schema_list":
        return "json_schema"

    return "normal"


def _is_descriptor_schema_root_path(path: str) -> bool:
    return re.fullmatch(
        r"AVAILABLE_AGENTS\[\d+\]\.(?:input_schema|output_schema)",
        path,
    ) is not None


def _is_secret_like_value(value: str) -> bool:
    stripped_value = value.strip()
    return bool(stripped_value) and any(
        pattern.search(stripped_value) for pattern in SECRET_VALUE_PATTERNS
    )


def _schema_map_path_segment(value: Any) -> str:
    value_text = str(value)
    if _is_secret_like_value(value_text):
        return REDACTED_SCHEMA_PATH_SEGMENT

    return value_text


def _validated_mapping_key_path_segment(path: str, key: Any) -> str:
    key_text = str(key)
    if _is_secret_like_value(key_text):
        raise DescriptorValidationError(
            f"descriptor config contains {_secret_like_mapping_key_label(path)}: "
            f"{path}.{REDACTED_SCHEMA_PATH_SEGMENT}"
        )

    return _safe_mapping_path_segment(path, key_text)


def _safe_mapping_path_segment(path: str, key_text: str) -> str:
    if _is_schema_map_container_path(path):
        return _schema_map_path_segment(key_text)

    return key_text


def _secret_like_mapping_key_label(path: str) -> str:
    if _is_json_schema_path(path):
        return "secret-like schema map key"

    return "secret-like mapping key"


def _is_json_schema_path(path: str) -> bool:
    return ".input_schema" in path or ".output_schema" in path


def _is_schema_map_container_path(path: str) -> bool:
    if not _is_json_schema_path(path):
        return False
    return any(
        path.endswith(f".{container_name}")
        for container_name in JSON_SCHEMA_REDACTED_KEY_CONTAINERS
    )


def _is_secret_like_field_name(field_name: str) -> bool:
    normalized = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", field_name)
    normalized = normalized.lower().replace("-", "_")
    compact_normalized = normalized.replace("_", "")
    return any(
        marker in normalized or marker.replace("_", "") in compact_normalized
        for marker in SECRET_FIELD_MARKERS
    )


def _safe_path_component(value: Any) -> str:
    if isinstance(value, int):
        return str(value)

    if not isinstance(value, str):
        return type(value).__name__

    if _is_secret_like_value(value):
        return "<redacted-key>"

    return value


def _redacted_validation_message(exc: ValidationError) -> str:
    messages: list[str] = []
    for error in exc.errors(include_url=False, include_input=False):
        location = ".".join(_safe_path_component(part) for part in error["loc"])
        messages.append(f"{location}: {error['msg']}")
    return "; ".join(messages)


__all__ = [
    "DescriptorValidationError",
    "REQUIRED_SPECIALIST_AGENT_IDS",
    "validate_agent_descriptors",
]
