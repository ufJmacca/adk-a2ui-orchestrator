"""Parse structured A2UI userAction events for plan approval surfaces."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from orchestrator_demo.a2a_support.transport import DataPart
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
    if isinstance(candidate.get("userAction"), Mapping):
        return _with_renderer_edit_state(candidate, candidate)

    action = candidate.get("action")
    if isinstance(action, Mapping):
        return _event_payload_from_mapping(_with_renderer_edit_state(action, candidate))

    derived_payload = _derive_user_action_from_event_name(candidate)
    if derived_payload is not None:
        return derived_payload

    event = candidate.get("event")
    if isinstance(event, Mapping):
        return _event_payload_from_event(event, candidate)

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


def _event_payload_from_event(
    event: Mapping[str, Any],
    source: Mapping[str, Any],
) -> Mapping[str, Any]:
    payload = _with_renderer_edit_state(event, source)
    derived_payload = _derive_user_action_from_event_name(payload)
    if derived_payload is not None:
        return derived_payload
    return _event_payload_from_mapping(payload)


def _derive_user_action_from_event_name(
    event: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    event_name = event.get("name")
    context = event.get("context")
    if not isinstance(event_name, str) or not isinstance(context, Mapping):
        return None

    if isinstance(context.get("type"), str):
        return None

    surface_id = context.get("surfaceId") or context.get("surface_id")
    if not isinstance(surface_id, str):
        return None

    action_payload = context.get("payload")
    context_payload = {
        key: value
        for key, value in context.items()
        if key not in {"payload"}
    }
    if isinstance(action_payload, Mapping):
        _reject_conflicting_event_metadata(context_payload, action_payload)
        payload = {
            **action_payload,
            **context_payload,
        }
    else:
        payload = {
            key: value
            for key, value in context_payload.items()
            if key not in {"surfaceId", "surface_id"}
        }

    return {
        "type": event_name,
        "surfaceId": surface_id,
        "payload": payload,
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
    return {
        key: value
        for key, value in payload.items()
        if key not in {"approvalEdits", "data", "values", "formData", "state"}
    }


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
    location = ".".join(str(part) for part in first_error.get("loc", ()))
    message = first_error.get("msg", str(exc))
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
