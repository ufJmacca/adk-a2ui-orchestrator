"""Parse structured A2UI userAction events for plan approval surfaces."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from orchestrator_demo.a2a_support.transport import DataPart
from orchestrator_demo.a2ui_support.secret_safety import (
    redact_secret_like_values,
    safe_path_component,
)
from orchestrator_demo.contracts import (
    PLAN_APPROVAL_SURFACE_PREFIX,
    PLAN_USER_ACTION_TYPES,
    UserAction,
)


SUPPORTED_PLAN_USER_ACTION_TYPES: set[str] = set(PLAN_USER_ACTION_TYPES)
_EVENT_METADATA_ALIASES: dict[str, tuple[str, ...]] = {
    "surfaceId": ("surfaceId", "surface_id"),
    "planId": ("planId", "plan_id"),
    "planVersion": (
        "planVersion",
        "plan_version",
        "editedPlanVersion",
        "edited_plan_version",
    ),
}
_USER_ACTION_FIELDS = {
    "actionId",
    "action_id",
    "type",
    "surfaceId",
    "surface_id",
    "planId",
    "plan_id",
    "planVersion",
    "plan_version",
    "payload",
}
_JSON_LITERAL_CONTEXT_KEYS = {
    "approvalEdits",
    "approvedStepIds",
    "editableFields",
    "filters",
    "orderedStepIds",
}


class StructuredUserActionRequiredError(ValueError):
    """Raised when input is not a structured A2UI userAction event."""


class PlanUserActionParseError(ValueError):
    """Raised when a structured userAction fails contract validation."""


class UnsupportedUserActionError(ValueError):
    """Raised when a userAction type is not supported for plan state."""


def parse_user_action(candidate: Any) -> UserAction:
    """Parse a structured A2UI userAction from a renderer event or DataPart."""

    event_payload = _extract_event_payload(candidate)
    try:
        return UserAction.model_validate(event_payload)
    except ValidationError as exc:
        raise PlanUserActionParseError(_validation_error_summary(exc)) from None


def parse_plan_user_action(candidate: Any) -> UserAction:
    """Parse and validate a supported plan-surface userAction event."""

    action = parse_user_action(candidate)
    if action.type not in SUPPORTED_PLAN_USER_ACTION_TYPES:
        raise UnsupportedUserActionError(
            f"unsupported plan userAction type: {action.type}"
        )
    if not action.surface_id.startswith(PLAN_APPROVAL_SURFACE_PREFIX):
        raise PlanUserActionParseError(
            "plan userAction events must target a plan approval surface"
        )
    if action.plan_id is None:
        raise PlanUserActionParseError("plan userAction events require planId")
    if action.type != "reject_plan" and action.plan_version is None:
        raise PlanUserActionParseError("plan userAction events require planVersion")

    return action


def _extract_event_payload(candidate: Any) -> Mapping[str, Any]:
    if isinstance(candidate, DataPart):
        return _event_payload_from_mapping(candidate.data)

    if isinstance(candidate, Mapping):
        return _event_payload_from_mapping(candidate)

    model_dump = getattr(candidate, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(by_alias=True, mode="json")
        if isinstance(dumped, Mapping):
            return _event_payload_from_mapping(dumped)

    raise StructuredUserActionRequiredError(
        "plan approval requires a structured A2UI userAction event"
    )


def _event_payload_from_mapping(
    candidate: Mapping[str, Any],
    source: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    source = candidate if source is None else source
    user_action = candidate.get("userAction")
    if isinstance(user_action, Mapping):
        derived_payload = _adk_rendered_user_action_payload(user_action, source)
        if derived_payload is not None:
            return derived_payload
        return _with_source_approval_edits(
            _direct_user_action_payload(user_action),
            source,
        )

    action = candidate.get("action")
    if isinstance(action, Mapping):
        return _event_payload_from_mapping(_with_renderer_edit_state(action, source), source)

    derived_payload = _derive_user_action_from_event_name(candidate)
    if derived_payload is not None:
        return _with_source_approval_edits(derived_payload, source)

    event = candidate.get("event")
    if isinstance(event, Mapping):
        return _event_payload_from_event(event, source)

    context = candidate.get("context")
    if isinstance(context, Mapping):
        return _event_payload_from_mapping(
            _with_renderer_edit_state(_decoded_mapping_context(context), source),
            source,
        )
    basic_catalog_context = _basic_catalog_payload_from(context)
    if basic_catalog_context is not None:
        return _event_payload_from_mapping(
            _with_renderer_edit_state(basic_catalog_context, source),
            source,
        )

    if isinstance(candidate.get("type"), str) and isinstance(
        candidate.get("surfaceId") or candidate.get("surface_id"),
        str,
    ):
        return _with_source_approval_edits(_direct_user_action_payload(candidate), source)

    data = candidate.get("data")
    if isinstance(data, Mapping):
        return _event_payload_from_mapping(_with_renderer_edit_state(data, source), source)

    raise StructuredUserActionRequiredError(
        "plan approval requires a structured A2UI userAction event"
    )


def _event_payload_from_event(
    event: Mapping[str, Any],
    source: Mapping[str, Any],
) -> Mapping[str, Any]:
    payload = _with_renderer_edit_state(event, source)
    derived_payload = _derive_user_action_from_event_name(payload)
    if derived_payload is not None:
        return _with_source_approval_edits(derived_payload, source)
    return _event_payload_from_mapping(payload, source)


def _derive_user_action_from_event_name(
    event: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    event_name = event.get("name")
    context_value = event.get("context")
    context = (
        _decoded_mapping_context(context_value)
        if isinstance(context_value, Mapping)
        else _basic_catalog_payload_from(context_value)
    )
    if not isinstance(event_name, str) or not isinstance(context, Mapping):
        return None

    if isinstance(context.get("type"), str):
        return None

    return _user_action_payload_from_event_name_context(event_name, context)


def _adk_rendered_user_action_payload(
    user_action: Mapping[str, Any],
    source: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    payload = _with_renderer_edit_state(user_action, source)
    event_name = payload.get("name")
    if not isinstance(event_name, str):
        return None

    context_value = payload.get("context")
    context = (
        _decoded_mapping_context(context_value)
        if isinstance(context_value, Mapping)
        else _basic_catalog_payload_from(context_value)
    )
    if not isinstance(context, Mapping):
        return None

    surface_id = (
        payload.get("surfaceId")
        or payload.get("surface_id")
        or context.get("surfaceId")
        or context.get("surface_id")
    )
    if not isinstance(surface_id, str):
        return None

    context_with_surface = {
        **context,
        "surfaceId": surface_id,
    }
    derived_payload = _user_action_payload_from_event_name_context(
        event_name,
        context_with_surface,
    )
    if derived_payload is None:
        return None
    derived_payload = _with_source_approval_edits(derived_payload, source)

    action_id = payload.get("actionId") or payload.get("action_id")
    if isinstance(action_id, str):
        return {
            **derived_payload,
            "actionId": action_id,
        }
    return derived_payload


def _user_action_payload_from_event_name_context(
    event_name: str,
    context: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    surface_id = context.get("surfaceId") or context.get("surface_id")
    if not isinstance(surface_id, str):
        return None
    action_type = context.get("type")
    if not isinstance(action_type, str):
        action_type = event_name

    action_payload = context.get("payload")
    context_payload = {
        key: value
        for key, value in context.items()
        if key not in {"payload"}
    }
    if isinstance(action_payload, Mapping):
        is_plan_action = (
            action_type in SUPPORTED_PLAN_USER_ACTION_TYPES
            or surface_id.startswith(PLAN_APPROVAL_SURFACE_PREFIX)
        )
        if is_plan_action:
            _reject_conflicting_event_metadata(context_payload, action_payload)
        excluded_metadata = {"type"}
        if not is_plan_action:
            excluded_metadata.update({"surfaceId", "surface_id"})
        payload = {
            **{
                key: value
                for key, value in context_payload.items()
                if key not in excluded_metadata
            },
            **action_payload,
        }
    else:
        payload = {
            key: value
            for key, value in context_payload.items()
            if key not in {"surfaceId", "surface_id"}
        }

    return {
        "type": action_type,
        "surfaceId": surface_id,
        "payload": payload,
    }


def _with_source_approval_edits(
    payload: Mapping[str, Any],
    source: Mapping[str, Any],
) -> Mapping[str, Any]:
    approval_edits = _approval_edits_from(source)
    action_payload = payload.get("payload")
    if (
        approval_edits is None
        or not isinstance(action_payload, Mapping)
        or "approvalEdits" in action_payload
    ):
        return payload

    return {
        **payload,
        "payload": {
            **action_payload,
            "approvalEdits": approval_edits,
        },
    }


def _reject_conflicting_event_metadata(
    context_payload: Mapping[str, Any],
    action_payload: Mapping[str, Any],
) -> None:
    for metadata_name, aliases in _EVENT_METADATA_ALIASES.items():
        context_values = _metadata_values(context_payload, aliases)
        action_values = _metadata_values(action_payload, aliases)
        all_values = [*context_values, *action_values]
        if not all_values:
            continue

        first_value = all_values[0]
        if any(value != first_value for value in all_values[1:]):
            raise PlanUserActionParseError(
                f"conflicting event metadata for {metadata_name}"
            )


def _metadata_values(
    payload: Mapping[str, Any],
    aliases: tuple[str, ...],
) -> list[Any]:
    return [payload[alias] for alias in aliases if payload.get(alias) is not None]


def _with_renderer_edit_state(
    payload: Mapping[str, Any],
    source: Mapping[str, Any],
) -> Mapping[str, Any]:
    approval_edits = _approval_edits_from(source)
    if approval_edits is None:
        return payload

    user_action = payload.get("userAction")
    if isinstance(user_action, Mapping):
        return {
            **payload,
            "userAction": _with_renderer_edit_state(user_action, source),
        }

    event = payload.get("event")
    if isinstance(event, Mapping):
        return {
            **payload,
            "event": _with_renderer_edit_state(event, source),
        }

    context = payload.get("context")
    if isinstance(context, Mapping):
        return {
            **payload,
            "context": _with_renderer_edit_state(context, source),
        }

    action_payload = payload.get("payload")
    if not isinstance(action_payload, Mapping) or "approvalEdits" in action_payload:
        return payload

    return {
        **payload,
        "payload": {
            **action_payload,
            "approvalEdits": approval_edits,
        },
    }


def _direct_user_action_payload(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = _with_renderer_edit_state(candidate, candidate)
    direct_payload = {
        key: value
        for key, value in payload.items()
        if key not in {"approvalEdits", "data", "values", "formData", "state"}
    }
    action_payload = direct_payload.get("payload")
    normalized_payload = _basic_catalog_payload_from(
        action_payload,
        decode_json_literals=True,
    )
    if normalized_payload is not None:
        direct_payload = {
            **direct_payload,
            "payload": normalized_payload,
        }
    elif not isinstance(action_payload, Mapping):
        direct_payload = _with_flattened_context_payload(direct_payload)
    return direct_payload


def _with_flattened_context_payload(
    direct_payload: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = dict(direct_payload)
    extra_payload = {
        key: _decoded_flattened_context_value(key, normalized.pop(key))
        for key in list(normalized)
        if key not in _USER_ACTION_FIELDS
    }
    if not extra_payload:
        return normalized

    for metadata_name in ("planId", "planVersion"):
        if metadata_name in normalized:
            extra_payload.setdefault(metadata_name, normalized[metadata_name])

    normalized["payload"] = extra_payload
    return normalized


def _basic_catalog_payload_from(
    value: Any,
    *,
    decode_json_literals: bool = False,
) -> dict[str, Any] | None:
    if not isinstance(value, list):
        return None

    payload: dict[str, Any] = {}
    for item in value:
        if not isinstance(item, Mapping):
            return None
        key = item.get("key")
        if not isinstance(key, str) or not key:
            return None
        payload[key] = _basic_catalog_value_from(
            item.get("value"),
            key=key,
            decode_json_literals=decode_json_literals or key == "payload",
        )
    return payload


def _basic_catalog_value_from(
    value: Any,
    *,
    key: str | None = None,
    decode_json_literals: bool = False,
) -> Any:
    nested_payload = _basic_catalog_payload_from(
        value,
        decode_json_literals=decode_json_literals,
    )
    if nested_payload is not None:
        return nested_payload
    if isinstance(value, Mapping):
        if isinstance(value.get("literalString"), str):
            return _decoded_literal_string(
                value["literalString"],
                key=key,
                decode_json_literal=decode_json_literals,
            )
        if isinstance(value.get("literalNumber"), int | float) and not isinstance(
            value.get("literalNumber"),
            bool,
        ):
            return value["literalNumber"]
        if isinstance(value.get("literalBoolean"), bool):
            return value["literalBoolean"]
        if isinstance(value.get("path"), str):
            return {"path": value["path"]}
    return value


def _decoded_flattened_context_value(key: str, value: Any) -> Any:
    if isinstance(value, str):
        return _decoded_literal_string(value, key=key)
    return value


def _decoded_mapping_context(context: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: _decoded_mapping_context_value(
            key,
            value,
            decode_json_literals=key == "payload",
        )
        for key, value in context.items()
    }


def _decoded_mapping_context_value(
    key: Any,
    value: Any,
    *,
    decode_json_literals: bool = False,
) -> Any:
    if isinstance(value, str):
        return _decoded_literal_string(
            value,
            key=key if isinstance(key, str) else None,
            decode_json_literal=decode_json_literals,
        )
    if isinstance(value, Mapping):
        return {
            nested_key: _decoded_mapping_context_value(
                nested_key,
                nested_value,
            )
            for nested_key, nested_value in value.items()
        }
    if isinstance(value, list):
        return [
            _decoded_mapping_context_value(
                None,
                item,
            )
            for item in value
        ]
    return value


def _decoded_literal_string(
    value: str,
    *,
    key: str | None,
    decode_json_literal: bool = False,
) -> Any:
    if not decode_json_literal and key not in _JSON_LITERAL_CONTEXT_KEYS:
        return value
    if value[:1] not in {"[", "{"}:
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _approval_edits_from(source: Mapping[str, Any]) -> Any | None:
    direct = source.get("approvalEdits")
    if direct is not None:
        return direct

    for container_name in ("data", "values", "formData", "state"):
        container = source.get(container_name)
        if isinstance(container, Mapping) and container.get("approvalEdits") is not None:
            return container["approvalEdits"]

    return None


def _validation_error_summary(exc: ValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return str(exc)

    first_error = errors[0]
    location = ".".join(
        safe_path_component(str(part)) for part in first_error.get("loc", ())
    )
    message = redact_secret_like_values(first_error.get("msg", str(exc)))
    if location:
        return f"{location}: {message}"
    return str(message)


__all__ = [
    "PlanUserActionParseError",
    "SUPPORTED_PLAN_USER_ACTION_TYPES",
    "StructuredUserActionRequiredError",
    "UnsupportedUserActionError",
    "parse_plan_user_action",
    "parse_user_action",
]
