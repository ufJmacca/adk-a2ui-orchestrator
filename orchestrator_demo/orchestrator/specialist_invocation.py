"""Shared specialist handler invocation utilities."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
import inspect
from typing import Any, Protocol

from orchestrator_demo.a2a_support.transport import DataPart, TextPart
from orchestrator_demo.a2ui_support.schema_manager import (
    CREATE_SURFACE_MESSAGE,
    DELETE_SURFACE_MESSAGE,
    UPDATE_COMPONENTS_MESSAGE,
    UPDATE_DATA_MODEL_MESSAGE,
)
from orchestrator_demo.a2ui_support.validation import (
    A2UIValidationResult,
    validate_outbound_a2ui,
)
from orchestrator_demo.contracts import SpecialistRequest, SpecialistResponse


SpecialistCallable = Callable[
    [SpecialistRequest],
    Awaitable[SpecialistResponse] | SpecialistResponse,
]


class SpecialistHandler(Protocol):
    """Handler-style specialist such as the local SpecialistAgent protocol."""

    def handle(
        self,
        request: SpecialistRequest,
    ) -> Awaitable[SpecialistResponse] | SpecialistResponse:
        """Handle one specialist request."""


SpecialistLike = SpecialistCallable | SpecialistHandler
SURFACE_ID_ENVELOPE_KEYS = (
    CREATE_SURFACE_MESSAGE,
    UPDATE_COMPONENTS_MESSAGE,
    UPDATE_DATA_MODEL_MESSAGE,
    DELETE_SURFACE_MESSAGE,
)


async def invoke_specialist(
    specialist: SpecialistLike,
    request: SpecialistRequest,
    *,
    enforce_response_agent_id: bool = True,
) -> SpecialistResponse:
    """Invoke callable specialists or SpecialistAgent-style handlers."""

    handler = getattr(specialist, "handle", None)
    if handler is None:
        handler = specialist
    if not callable(handler):
        raise RuntimeError(
            f"specialist handler for {request.agent_id!r} is not callable"
        )

    result = handler(request)
    if inspect.isawaitable(result):
        result = await result

    response = SpecialistResponse.model_validate(result)
    if enforce_response_agent_id and response.agent_id != request.agent_id:
        raise RuntimeError(
            "specialist response agent_id must match requested agent_id: "
            f"expected {request.agent_id!r}, got {response.agent_id!r}"
        )
    return _validated_a2ui_response(response)


def _validated_a2ui_response(response: SpecialistResponse) -> SpecialistResponse:
    if response.a2ui_payload is None:
        return response

    validation = validate_outbound_a2ui(response.a2ui_payload)
    if validation.valid:
        payloads = [
            part.data
            for part in validation.renderer_parts
            if isinstance(part, DataPart)
        ]
        if len(payloads) != len(validation.renderer_parts):
            return _a2ui_fallback_response(response, validation)

        a2ui_payload = (
            payloads
            if isinstance(response.a2ui_payload, list) or len(payloads) != 1
            else payloads[0]
        )
        payload_surface_id, surface_error = _surface_id_from_payloads(payloads)
        if surface_error is not None:
            return _a2ui_fallback_response(
                response,
                _a2ui_surface_validation_error(surface_error),
            )
        return response.model_copy(
            update={
                "a2ui_payload": a2ui_payload,
                "surface_id": payload_surface_id or response.surface_id,
            }
        )

    return _a2ui_fallback_response(response, validation)


def _surface_id_from_payloads(
    payloads: Sequence[dict[str, Any]],
) -> tuple[str | None, str | None]:
    surface_ids: set[str] = set()
    for payload in payloads:
        surface_id = _surface_id_from_payload(payload)
        if surface_id is not None:
            surface_ids.add(surface_id)
    if len(surface_ids) > 1:
        return (
            None,
            "A2UI response envelopes must target one surfaceId; "
            "found multiple surfaceIds",
        )
    if surface_ids:
        return next(iter(surface_ids)), None
    return None, None


def _surface_id_from_payload(payload: Mapping[str, Any]) -> str | None:
    surface_id = payload.get("surfaceId")
    if isinstance(surface_id, str) and surface_id:
        return surface_id

    for envelope_key in SURFACE_ID_ENVELOPE_KEYS:
        envelope = payload.get(envelope_key)
        if isinstance(envelope, Mapping):
            surface_id = envelope.get("surfaceId")
            if isinstance(surface_id, str) and surface_id:
                return surface_id

    return None


def _a2ui_fallback_response(
    response: SpecialistResponse,
    validation: A2UIValidationResult,
) -> SpecialistResponse:
    diagnostic = {
        "valid": False,
        "validationErrors": list(validation.validation_errors),
        "fallbackPart": validation.renderer_part.model_dump(
            mode="json",
            exclude_none=True,
        ),
    }
    structured_output = {
        **response.structured_output,
        "a2ui_validation": diagnostic,
    }
    return response.model_copy(
        update={
            "structured_output": structured_output,
            "a2ui_payload": None,
            "surface_id": None,
        }
    )


def _a2ui_surface_validation_error(error: str) -> A2UIValidationResult:
    diagnostic = {
        "fallback": "text",
        "validationErrors": [error],
        "repairAttempted": False,
    }
    return A2UIValidationResult(
        valid=False,
        renderer_part=TextPart(
            text=(
                "A2UI rendering unavailable. The generated UI payload failed "
                "validation and was not emitted to the renderer."
            ),
            metadata={"developerDiagnostic": diagnostic},
        ),
        validation_errors=[error],
        repaired=False,
    )


__all__ = [
    "SpecialistCallable",
    "SpecialistHandler",
    "SpecialistLike",
    "invoke_specialist",
]
