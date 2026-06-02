"""Parse structured A2UI userAction events for plan approval surfaces."""

from __future__ import annotations

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


def _event_payload_from_mapping(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    user_action = candidate.get("userAction")
    if isinstance(user_action, Mapping):
        return _direct_user_action_payload(
            _with_renderer_edit_state(user_action, candidate)
        )

    action = candidate.get("action")
    if isinstance(action, Mapping):
        return _event_payload_from_mapping(_with_renderer_edit_state(action, candidate))

    event = candidate.get("event")
    if isinstance(event, Mapping):
        return _event_payload_from_mapping(_with_renderer_edit_state(event, candidate))

    context = candidate.get("context")
    if isinstance(context, Mapping):
        return _event_payload_from_mapping(_with_renderer_edit_state(context, candidate))

    if isinstance(candidate.get("type"), str) and isinstance(
        candidate.get("surfaceId") or candidate.get("surface_id"),
        str,
    ):
        return _direct_user_action_payload(candidate)

    data = candidate.get("data")
    if isinstance(data, Mapping):
        return _event_payload_from_mapping(_with_renderer_edit_state(data, candidate))

    raise StructuredUserActionRequiredError(
        "plan approval requires a structured A2UI userAction event"
    )


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
    normalized_payload = _basic_catalog_payload_from(action_payload)
    if normalized_payload is not None:
        direct_payload = {
            **direct_payload,
            "payload": normalized_payload,
        }
    return direct_payload


def _basic_catalog_payload_from(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, list):
        return None

    payload: dict[str, Any] = {}
    for item in value:
        if not isinstance(item, Mapping):
            return None
        key = item.get("key")
        if not isinstance(key, str) or not key:
            return None
        payload[key] = item.get("value")
    return payload


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
