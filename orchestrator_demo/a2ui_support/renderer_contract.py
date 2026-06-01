"""Renderer-facing A2UI pass-through helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from orchestrator_demo.a2a_support.transport import DataPart
from orchestrator_demo.a2ui_support.schema_manager import (
    CREATE_SURFACE_MESSAGE,
    DELETE_SURFACE_MESSAGE,
    UPDATE_COMPONENTS_MESSAGE,
    UPDATE_DATA_MODEL_MESSAGE,
)
from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui
from orchestrator_demo.contracts import A2uiPayload
from orchestrator_demo.orchestrator.surface_routes import SurfaceRouteRegistry


class RendererContractError(ValueError):
    """Raised when an A2UI payload cannot be safely emitted to the renderer."""


def prepare_specialist_a2ui_for_renderer(
    payload: A2uiPayload | DataPart | Sequence[DataPart],
    *,
    owner_agent_id: str,
    surface_registry: SurfaceRouteRegistry,
) -> list[DataPart]:
    """Validate specialist A2UI, preserve it unchanged, and register ownership."""

    parts = _validated_data_parts(payload)
    staged_owners = dict(surface_registry._owners_by_surface_id)
    for part in parts:
        for surface_id in deleted_surface_ids_from_a2ui_payload(part.data):
            surface_registry._clear_specialist_surface_from(
                staged_owners, surface_id, agent_id=owner_agent_id
            )
        for surface_id in surface_ids_from_a2ui_payload(part.data):
            surface_registry._register_owner(
                staged_owners,
                surface_registry._specialist_surface_owner(
                    surface_id,
                    agent_id=owner_agent_id,
                ),
            )

    surface_registry._owners_by_surface_id = staged_owners

    return parts


def prepare_approval_a2ui_for_renderer(
    payload: A2uiPayload | DataPart | Sequence[DataPart],
    *,
    plan_id: str,
    surface_registry: SurfaceRouteRegistry,
) -> list[DataPart]:
    """Validate approval A2UI and register approval surfaces to the orchestrator."""

    parts = _validated_data_parts(payload)
    staged_owners = dict(surface_registry._owners_by_surface_id)
    for part in parts:
        for surface_id in deleted_surface_ids_from_a2ui_payload(part.data):
            surface_registry._clear_surface_from(
                staged_owners,
                surface_id,
                owner_type="orchestrator",
                owner_id="orchestrator",
            )
        for surface_id in surface_ids_from_a2ui_payload(part.data):
            surface_registry._register_owner(
                staged_owners,
                surface_registry._orchestrator_surface_owner(
                    surface_id,
                    plan_id=plan_id,
                    source="approval_surface",
                ),
            )

    surface_registry._owners_by_surface_id = staged_owners

    return parts


def surface_ids_from_a2ui_payload(payload: Mapping[str, Any]) -> list[str]:
    """Extract active surface ids from a validated A2UI server-to-client message."""

    surface_ids: list[str] = []
    for message_type in (
        CREATE_SURFACE_MESSAGE,
        UPDATE_COMPONENTS_MESSAGE,
        UPDATE_DATA_MODEL_MESSAGE,
    ):
        message = payload.get(message_type)
        if not isinstance(message, Mapping):
            continue
        surface_id = message.get("surfaceId")
        if isinstance(surface_id, str):
            surface_ids.append(surface_id)
    return surface_ids


def deleted_surface_ids_from_a2ui_payload(payload: Mapping[str, Any]) -> list[str]:
    """Extract deleted surface ids from a validated A2UI server-to-client message."""

    message = payload.get(DELETE_SURFACE_MESSAGE)
    if not isinstance(message, Mapping):
        return []
    surface_id = message.get("surfaceId")
    return [surface_id] if isinstance(surface_id, str) else []


def _validated_data_parts(
    payload: A2uiPayload | DataPart | Sequence[DataPart],
) -> list[DataPart]:
    parts: list[DataPart] = []
    for candidate in _iter_payload_messages(payload):
        result = validate_outbound_a2ui(candidate)
        if not isinstance(result.renderer_part, DataPart):
            errors = "; ".join(result.validation_errors)
            raise RendererContractError(
                "A2UI payload failed validation and was not emitted to the "
                f"renderer: {errors}"
            )
        parts.append(result.renderer_part)

    return parts


def _iter_payload_messages(
    payload: A2uiPayload | DataPart | Sequence[DataPart],
) -> list[Any]:
    if isinstance(payload, DataPart):
        return [payload]
    if isinstance(payload, Mapping):
        return [payload]
    return list(payload)


__all__ = [
    "RendererContractError",
    "deleted_surface_ids_from_a2ui_payload",
    "prepare_approval_a2ui_for_renderer",
    "prepare_specialist_a2ui_for_renderer",
    "surface_ids_from_a2ui_payload",
]
