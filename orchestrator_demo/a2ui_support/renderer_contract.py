"""Renderer-facing A2UI pass-through helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from orchestrator_demo.a2a_support.transport import DataPart, TextPart
from orchestrator_demo.a2ui_support.schema_manager import (
    CREATE_SURFACE_MESSAGE,
    DELETE_SURFACE_MESSAGE,
    UPDATE_COMPONENTS_MESSAGE,
    UPDATE_DATA_MODEL_MESSAGE,
)
from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui
from orchestrator_demo.contracts import A2uiPayload
from orchestrator_demo.orchestrator.surface_routes import (
    SurfaceOwnershipError,
    SurfaceOwnerType,
    SurfaceRouteRegistry,
)


class RendererContractError(ValueError):
    """Raised when an A2UI payload cannot be safely emitted to the renderer."""


def prepare_specialist_a2ui_for_renderer(
    payload: A2uiPayload | DataPart | Sequence[DataPart],
    *,
    owner_agent_id: str,
    surface_registry: SurfaceRouteRegistry,
) -> list[DataPart | TextPart]:
    """Validate specialist A2UI, preserve it unchanged, and register ownership."""

    parts = _validated_data_parts(payload)
    staged_owners = dict(surface_registry._owners_by_surface_id)
    for part in parts:
        if not isinstance(part, DataPart):
            continue
        for surface_id in deleted_surface_ids_from_a2ui_payload(part.data):
            _clear_surface_owned_by(
                staged_owners,
                surface_id,
                surface_registry=surface_registry,
                owner_type="specialist",
                owner_id=owner_agent_id,
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
) -> list[DataPart | TextPart]:
    """Validate approval A2UI and register approval surfaces to the orchestrator."""

    parts = _validated_data_parts(payload)
    staged_owners = dict(surface_registry._owners_by_surface_id)
    for part in parts:
        if not isinstance(part, DataPart):
            continue
        for surface_id in deleted_surface_ids_from_a2ui_payload(part.data):
            _clear_surface_owned_by(
                staged_owners,
                surface_id,
                surface_registry=surface_registry,
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
) -> list[DataPart | TextPart]:
    parts: list[DataPart | TextPart] = []
    for candidate in _iter_payload_messages(payload):
        result = validate_outbound_a2ui(candidate)
        parts.append(result.renderer_part)

    return parts


def _clear_surface_owned_by(
    owners_by_surface_id: dict[str, Any],
    surface_id: str,
    *,
    surface_registry: SurfaceRouteRegistry,
    owner_type: SurfaceOwnerType,
    owner_id: str,
) -> None:
    existing = owners_by_surface_id.get(surface_id)
    if existing is None:
        raise SurfaceOwnershipError(
            f"deleteSurface target {surface_id} is not registered to "
            f"{owner_type}:{owner_id}"
        )
    if existing.owner_type != owner_type or existing.owner_id != owner_id:
        raise SurfaceOwnershipError(
            f"deleteSurface target {surface_id} is owned by "
            f"{existing.owner_type}:{existing.owner_id}, not {owner_type}:{owner_id}"
        )
    surface_registry._clear_surface_from(
        owners_by_surface_id,
        surface_id,
        owner_type=owner_type,
        owner_id=owner_id,
    )


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
