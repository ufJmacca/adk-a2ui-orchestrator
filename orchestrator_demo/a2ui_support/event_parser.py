"""Parse structured A2UI userAction events for plan approval surfaces."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from orchestrator_demo.a2a_support.transport import DataPart
from orchestrator_demo.contracts import (
    PLAN_APPROVAL_SURFACE_PREFIX,
    UserAction,
)


SUPPORTED_PLAN_USER_ACTION_TYPES: set[str] = {
    "approve_plan",
    "reject_plan",
    "edit_plan",
    "remove_step",
    "reorder_steps",
    "replace_agent",
    "add_instruction",
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
        return candidate

    data = candidate.get("data")
    if isinstance(data, Mapping):
        return _event_payload_from_mapping(data)

    if isinstance(candidate.get("type"), str) and isinstance(
        candidate.get("surfaceId") or candidate.get("surface_id"),
        str,
    ):
        return candidate

    raise StructuredUserActionRequiredError(
        "plan approval requires a structured A2UI userAction event"
    )


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
