"""Deterministic A2UI surface ownership and userAction routing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from inspect import isawaitable
from typing import Any, Literal

from orchestrator_demo.a2ui_support.event_parser import (
    PlanUserActionParseError,
    StructuredUserActionRequiredError,
    parse_user_action,
)
from orchestrator_demo.a2ui_support.schema_manager import (
    CREATE_SURFACE_MESSAGE,
    DELETE_SURFACE_MESSAGE,
    PLAN_APPROVAL_SURFACE_PREFIX,
    SURFACE_ID_PATTERN,
    UPDATE_COMPONENTS_MESSAGE,
    UPDATE_DATA_MODEL_MESSAGE,
)
from orchestrator_demo.a2ui_support.validation import (
    SurfaceComponentGraphs,
    apply_validated_a2ui_component_graph,
    clone_surface_component_graphs,
    validate_outbound_a2ui,
)
from orchestrator_demo.app.logging import log_audit_event


SurfaceOwnerType = Literal["orchestrator", "specialist"]
RouteStatus = Literal["orchestrator_owned", "forwarded", "error"]


class SurfaceOwnershipError(ValueError):
    """Raised when surface ownership cannot be registered safely."""


@dataclass(frozen=True)
class SurfaceOwner:
    """Registered owner for one renderer surface."""

    surface_id: str
    owner_type: SurfaceOwnerType
    owner_id: str
    plan_id: str | None = None
    source: str | None = None


@dataclass(frozen=True)
class SurfaceRouteResult:
    """Result of routing a structured A2UI userAction."""

    status: RouteStatus
    surface_id: str | None
    owner: SurfaceOwner | None = None
    response: Any | None = None
    error: dict[str, Any] | None = None
    original_payload: Any | None = None


@dataclass(frozen=True)
class SurfaceRegistrySnapshot:
    """Snapshot of renderer surface ownership and component validation state."""

    owners_by_surface_id: dict[str, SurfaceOwner]
    components_by_surface_id: SurfaceComponentGraphs


class SurfaceRouteRegistry:
    """Track A2UI surface ownership and route future UI events by surfaceId."""

    def __init__(self) -> None:
        self._owners_by_surface_id: dict[str, SurfaceOwner] = {}
        self._components_by_surface_id: SurfaceComponentGraphs = {}

    def register_approval_surface(
        self,
        surface_id: str,
        *,
        plan_id: str,
    ) -> SurfaceOwner:
        """Register a plan approval surface as orchestrator-owned."""

        return self.register_orchestrator_surface(
            surface_id,
            plan_id=plan_id,
            source="approval_surface",
        )

    def register_orchestrator_surface(
        self,
        surface_id: str,
        *,
        plan_id: str | None = None,
        owner_id: str = "orchestrator",
        source: str | None = None,
    ) -> SurfaceOwner:
        """Register a surface owned by the orchestrator."""

        return self._register(
            self._orchestrator_surface_owner(
                surface_id,
                plan_id=plan_id,
                owner_id=owner_id,
                source=source,
            )
        )

    def register_specialist_surface(
        self,
        surface_id: str,
        *,
        agent_id: str,
        source: str | None = "specialist_a2ui",
    ) -> SurfaceOwner:
        """Register a surface owned by a specialist agent."""

        return self._register(
            self._specialist_surface_owner(
                surface_id,
                agent_id=agent_id,
                source=source,
            )
        )

    def owner_for(self, surface_id: str) -> SurfaceOwner | None:
        """Return the registered owner for a surface, if known."""

        return self._owners_by_surface_id.get(surface_id)

    def clear_surface(
        self,
        surface_id: str,
        *,
        owner_type: SurfaceOwnerType | None = None,
        owner_id: str | None = None,
    ) -> SurfaceOwner | None:
        """Remove ownership for a deleted surface when the expected owner matches."""

        removed = self._clear_surface_from(
            self._owners_by_surface_id,
            surface_id,
            owner_type=owner_type,
            owner_id=owner_id,
        )
        if removed is not None:
            self._components_by_surface_id.pop(surface_id, None)
        return removed

    def clear_all(self) -> SurfaceRegistrySnapshot:
        """Remove all registered renderer surface ownership and return a snapshot."""

        snapshot = SurfaceRegistrySnapshot(
            owners_by_surface_id=dict(self._owners_by_surface_id),
            components_by_surface_id=clone_surface_component_graphs(
                self._components_by_surface_id
            ),
        )
        self._owners_by_surface_id.clear()
        self._components_by_surface_id.clear()
        return snapshot

    def restore_all(
        self,
        snapshot: Mapping[str, SurfaceOwner] | SurfaceRegistrySnapshot,
    ) -> None:
        """Restore a previously captured renderer surface ownership snapshot."""

        if isinstance(snapshot, SurfaceRegistrySnapshot):
            self._owners_by_surface_id = dict(snapshot.owners_by_surface_id)
            self._components_by_surface_id = clone_surface_component_graphs(
                snapshot.components_by_surface_id
            )
            return

        self._owners_by_surface_id = dict(snapshot)

    def export_snapshot(self) -> dict[str, Any]:
        """Return JSON-safe surface ownership and validated component state."""

        return {
            "ownersBySurfaceId": {
                surface_id: _surface_owner_to_snapshot(owner)
                for surface_id, owner in self._owners_by_surface_id.items()
            },
            "componentsBySurfaceId": clone_surface_component_graphs(
                self._components_by_surface_id
            ),
        }

    def restore_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        """Restore JSON-safe surface ownership and validated component state."""

        owners_snapshot = snapshot.get("ownersBySurfaceId")
        if not isinstance(owners_snapshot, Mapping):
            raise SurfaceOwnershipError("ownersBySurfaceId must be an object")

        restored_owners: dict[str, SurfaceOwner] = {}
        for surface_id, owner_snapshot in owners_snapshot.items():
            if not isinstance(surface_id, str):
                raise SurfaceOwnershipError("surface owner keys must be strings")
            owner = _surface_owner_from_snapshot(owner_snapshot)
            if owner.surface_id != surface_id:
                raise SurfaceOwnershipError(
                    "surface owner key must match owner surfaceId"
                )
            restored_owners[surface_id] = owner

        components_snapshot = snapshot.get("componentsBySurfaceId", {})
        if not isinstance(components_snapshot, Mapping):
            raise SurfaceOwnershipError("componentsBySurfaceId must be an object")

        self._owners_by_surface_id = restored_owners
        self._components_by_surface_id = clone_surface_component_graphs(
            components_snapshot
        )

    def _clear_surface_from(
        self,
        owners_by_surface_id: dict[str, SurfaceOwner],
        surface_id: str,
        *,
        owner_type: SurfaceOwnerType | None = None,
        owner_id: str | None = None,
    ) -> SurfaceOwner | None:
        surface_id = _validated_surface_id(surface_id)
        existing = owners_by_surface_id.get(surface_id)
        if existing is None:
            return None
        if owner_type is not None and existing.owner_type != owner_type:
            return None
        if owner_id is not None and existing.owner_id != owner_id:
            return None

        return owners_by_surface_id.pop(surface_id)

    async def route_user_action(
        self,
        candidate: Any,
        *,
        specialist_adapters: Mapping[str, Any],
    ) -> SurfaceRouteResult:
        """Route a renderer userAction without LLM owner inference."""

        try:
            action = parse_user_action(candidate)
        except (PlanUserActionParseError, StructuredUserActionRequiredError) as exc:
            return _logged_route_result(
                SurfaceRouteResult(
                    status="error",
                    surface_id=None,
                    error=_routing_error(
                        code="invalid_user_action",
                        surface_id=None,
                        message=str(exc),
                    ),
                )
            )

        owner = self.owner_for(action.surface_id)
        if owner is None:
            return _logged_route_result(
                SurfaceRouteResult(
                    status="error",
                    surface_id=None,
                    error=_routing_error(
                        code="unknown_surface",
                        surface_id=None,
                        message=(
                            "No owner is registered for the requested A2UI surface."
                        ),
                    ),
                )
            )

        if owner.owner_type == "orchestrator":
            return _logged_route_result(
                SurfaceRouteResult(
                    status="orchestrator_owned",
                    surface_id=action.surface_id,
                    owner=owner,
                    original_payload=candidate,
                )
            )

        adapter = specialist_adapters.get(owner.owner_id)
        handler = getattr(adapter, "handle_user_action", None)
        if adapter is None or not callable(handler):
            return _logged_route_result(
                SurfaceRouteResult(
                    status="error",
                    surface_id=action.surface_id,
                    owner=owner,
                    error=_routing_error(
                        code="owner_unavailable",
                        surface_id=action.surface_id,
                        message=(
                            "No userAction handler is available for A2UI surface "
                            f"{action.surface_id} owner {owner.owner_id}."
                        ),
                    ),
                    original_payload=candidate,
                )
            )

        try:
            response = handler(candidate)
            if isawaitable(response):
                response = await response
        except Exception as exc:
            return _logged_route_result(
                SurfaceRouteResult(
                    status="error",
                    surface_id=action.surface_id,
                    owner=owner,
                    error=_routing_error(
                        code="owner_handler_failed",
                        surface_id=action.surface_id,
                        message=_owner_handler_failure_message(exc),
                    ),
                    original_payload=candidate,
                )
            )

        try:
            self._register_specialist_response_surfaces(response, owner=owner)
        except SurfaceOwnershipError as exc:
            return _logged_route_result(
                SurfaceRouteResult(
                    status="error",
                    surface_id=action.surface_id,
                    owner=owner,
                    error=_routing_error(
                        code="surface_registration_rejected",
                        surface_id=action.surface_id,
                        message=str(exc),
                    ),
                    original_payload=candidate,
                )
            )

        return _logged_route_result(
            SurfaceRouteResult(
                status="forwarded",
                surface_id=action.surface_id,
                owner=owner,
                response=response,
                original_payload=candidate,
            )
        )

    def _register(self, owner: SurfaceOwner) -> SurfaceOwner:
        return self._register_owner(self._owners_by_surface_id, owner)

    def _register_owner(
        self,
        owners_by_surface_id: dict[str, SurfaceOwner],
        owner: SurfaceOwner,
    ) -> SurfaceOwner:
        existing = owners_by_surface_id.get(owner.surface_id)
        if existing is not None and (
            existing.owner_type != owner.owner_type
            or existing.owner_id != owner.owner_id
            or existing.plan_id != owner.plan_id
        ):
            raise SurfaceOwnershipError(
                f"surface {owner.surface_id} is already owned by "
                f"{existing.owner_type}:{existing.owner_id}"
            )

        owners_by_surface_id[owner.surface_id] = owner
        return owner

    def _specialist_surface_owner(
        self,
        surface_id: str,
        *,
        agent_id: str,
        source: str | None = "specialist_a2ui",
    ) -> SurfaceOwner:
        if not agent_id.strip():
            raise SurfaceOwnershipError("specialist surface owner requires agent_id")

        surface_id = _validated_surface_id(surface_id)
        if surface_id.startswith(PLAN_APPROVAL_SURFACE_PREFIX):
            raise SurfaceOwnershipError(
                f"surface {surface_id} uses reserved approval surface prefix "
                f"{PLAN_APPROVAL_SURFACE_PREFIX}"
            )

        return SurfaceOwner(
            surface_id=surface_id,
            owner_type="specialist",
            owner_id=agent_id,
            source=source,
        )

    def _orchestrator_surface_owner(
        self,
        surface_id: str,
        *,
        plan_id: str | None = None,
        owner_id: str = "orchestrator",
        source: str | None = None,
    ) -> SurfaceOwner:
        return SurfaceOwner(
            surface_id=_validated_surface_id(surface_id),
            owner_type="orchestrator",
            owner_id=owner_id,
            plan_id=plan_id,
            source=source,
        )

    def _register_specialist_response_surfaces(
        self,
        response: Any,
        *,
        owner: SurfaceOwner,
    ) -> None:
        payload = _response_a2ui_payload(response)
        if payload is None:
            return

        validated_payloads: list[Mapping[str, Any]] = []
        staged_components = clone_surface_component_graphs(
            self._components_by_surface_id
        )
        for candidate in _iter_a2ui_messages(payload):
            result = validate_outbound_a2ui(
                candidate,
                existing_components_by_surface_id=staged_components,
            )
            if not result.valid:
                return
            data = getattr(result.renderer_part, "data", None)
            if not isinstance(data, Mapping):
                return
            validated_payloads.append(data)
            apply_validated_a2ui_component_graph(staged_components, data)

        staged_owners = dict(self._owners_by_surface_id)
        for data in validated_payloads:
            for surface_id in _deleted_surface_ids_from_validated_a2ui(data):
                self._clear_specialist_owned_surface(
                    staged_owners, surface_id, owner=owner
                )
            for surface_id in _surface_ids_from_validated_a2ui(data):
                self._register_owner(
                    staged_owners,
                    self._specialist_surface_owner(
                        surface_id,
                        agent_id=owner.owner_id,
                    ),
                )

        self._owners_by_surface_id = staged_owners
        self._components_by_surface_id = staged_components

    def _clear_specialist_owned_surface(
        self,
        owners_by_surface_id: dict[str, SurfaceOwner],
        surface_id: str,
        *,
        owner: SurfaceOwner,
    ) -> SurfaceOwner | None:
        return self._clear_specialist_surface_from(
            owners_by_surface_id,
            surface_id,
            agent_id=owner.owner_id,
        )

    def _clear_specialist_surface_from(
        self,
        owners_by_surface_id: dict[str, SurfaceOwner],
        surface_id: str,
        *,
        agent_id: str,
    ) -> SurfaceOwner | None:
        surface_id = _validated_surface_id(surface_id)
        existing = owners_by_surface_id.get(surface_id)
        if existing is not None and (
            existing.owner_type != "specialist" or existing.owner_id != agent_id
        ):
            raise SurfaceOwnershipError(
                f"specialist {agent_id} cannot delete surface {surface_id} "
                f"owned by {existing.owner_type}:{existing.owner_id}"
            )
        if existing is None and surface_id.startswith(PLAN_APPROVAL_SURFACE_PREFIX):
            raise SurfaceOwnershipError(
                f"specialist {agent_id} cannot delete reserved approval "
                f"surface {surface_id}"
            )

        return self._clear_surface_from(
            owners_by_surface_id,
            surface_id,
            owner_type="specialist",
            owner_id=agent_id,
        )


def _validated_surface_id(surface_id: str) -> str:
    if not SURFACE_ID_PATTERN.fullmatch(surface_id):
        raise SurfaceOwnershipError(f"invalid A2UI surfaceId: {surface_id}")
    return surface_id


def _routing_error(
    *,
    code: str,
    surface_id: str | None,
    message: str,
) -> dict[str, Any]:
    return {
        "code": code,
        "surfaceId": surface_id,
        "message": message,
        "ownerInferenceAttempted": False,
    }


def _logged_route_result(result: SurfaceRouteResult) -> SurfaceRouteResult:
    _log_ui_event_routed(result)
    return result


def _log_ui_event_routed(result: SurfaceRouteResult) -> None:
    owner = result.owner
    error = result.error or {}
    log_audit_event(
        "ui_event_routed",
        {
            "status": result.status,
            "surface_id": result.surface_id,
            "owner_type": owner.owner_type if owner is not None else None,
            "owner_id": owner.owner_id if owner is not None else None,
            "error_code": error.get("code"),
            "owner_inference_attempted": bool(
                error.get("ownerInferenceAttempted", False)
            ),
        },
    )


def _owner_handler_failure_message(exc: Exception) -> str:
    return (
        "A2UI surface owner handler failed: "
        f"{type(exc).__name__}. Error details redacted."
    )


def _response_a2ui_payload(response: Any) -> Any | None:
    if isinstance(response, Mapping):
        return response.get("a2ui_payload") or response.get("a2uiPayload")
    return getattr(response, "a2ui_payload", None)


def _iter_a2ui_messages(payload: Any) -> list[Any]:
    if isinstance(payload, Mapping):
        return [payload]
    if isinstance(payload, list | tuple):
        return list(payload)
    return [payload]


def _surface_ids_from_validated_a2ui(payload: Mapping[str, Any]) -> list[str]:
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


def _deleted_surface_ids_from_validated_a2ui(payload: Mapping[str, Any]) -> list[str]:
    message = payload.get(DELETE_SURFACE_MESSAGE)
    if not isinstance(message, Mapping):
        return []
    surface_id = message.get("surfaceId")
    return [surface_id] if isinstance(surface_id, str) else []


def _surface_owner_to_snapshot(owner: SurfaceOwner) -> dict[str, Any]:
    return {
        "surfaceId": owner.surface_id,
        "ownerType": owner.owner_type,
        "ownerId": owner.owner_id,
        "planId": owner.plan_id,
        "source": owner.source,
    }


def _surface_owner_from_snapshot(candidate: Any) -> SurfaceOwner:
    if not isinstance(candidate, Mapping):
        raise SurfaceOwnershipError("surface owner snapshot must be an object")

    owner_type = candidate.get("ownerType")
    if owner_type not in {"orchestrator", "specialist"}:
        raise SurfaceOwnershipError("surface owner snapshot has invalid ownerType")

    surface_id = candidate.get("surfaceId")
    owner_id = candidate.get("ownerId")
    if not isinstance(surface_id, str) or not isinstance(owner_id, str):
        raise SurfaceOwnershipError(
            "surface owner snapshot requires surfaceId and ownerId"
        )

    plan_id = _optional_snapshot_string(candidate.get("planId"))
    source = _optional_snapshot_string(candidate.get("source"))
    return SurfaceOwner(
        surface_id=_validated_surface_id(surface_id),
        owner_type=owner_type,
        owner_id=owner_id,
        plan_id=plan_id,
        source=source,
    )


def _optional_snapshot_string(candidate: Any) -> str | None:
    if candidate is None:
        return None
    if isinstance(candidate, str):
        return candidate
    raise SurfaceOwnershipError("surface owner snapshot field must be a string or null")


__all__ = [
    "RouteStatus",
    "SurfaceOwner",
    "SurfaceOwnerType",
    "SurfaceOwnershipError",
    "SurfaceRouteRegistry",
    "SurfaceRouteResult",
    "SurfaceRegistrySnapshot",
]
