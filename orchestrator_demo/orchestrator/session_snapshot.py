"""JSON-safe orchestrator session snapshot primitives."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from orchestrator_demo.a2ui_support.secret_safety import (
    redact_secret_like_values,
    safe_path_component,
)
from orchestrator_demo.contracts import AgentDescriptor, ExecutionPlan
from orchestrator_demo.orchestrator.approval_state import ApprovalStateStore
from orchestrator_demo.orchestrator.graph_runtime import GraphRuntime
from orchestrator_demo.orchestrator.request_context import RequestContext
from orchestrator_demo.orchestrator.surface_routes import SurfaceRouteRegistry


SNAPSHOT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RestoredSessionSnapshot:
    """State containers rehydrated from one orchestrator session snapshot."""

    approval_store: ApprovalStateStore
    request_contexts_by_plan_id: dict[str, RequestContext]
    surface_registry: SurfaceRouteRegistry
    artifact_refs: dict[str, Any]


def export_session_snapshot(
    *,
    approval_store: ApprovalStateStore,
    request_contexts_by_plan_id: Mapping[str, RequestContext],
    surface_registry: SurfaceRouteRegistry,
    artifact_refs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a versioned, JSON-safe snapshot of pending orchestrator state."""

    snapshot = {
        "schemaVersion": SNAPSHOT_SCHEMA_VERSION,
        "approvalRecords": approval_store.export_records(),
        "requestContextsByPlanId": {
            plan_id: context.export_snapshot()
            for plan_id, context in request_contexts_by_plan_id.items()
        },
        "surfaceRegistry": surface_registry.export_snapshot(),
        "artifactRefs": dict(artifact_refs or {}),
    }
    return _redacted_json_safe(snapshot)


def restore_session_snapshot(
    snapshot: Mapping[str, Any],
    *,
    agent_descriptors: Sequence[AgentDescriptor]
    | Callable[[], Sequence[AgentDescriptor]],
    graph_runtime: GraphRuntime | None = None,
    plan_validator: Callable[[ExecutionPlan], None] | None = None,
) -> RestoredSessionSnapshot:
    """Rehydrate state containers from a versioned session snapshot."""

    schema_version = snapshot.get("schemaVersion")
    if schema_version != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(
            "unsupported orchestrator session snapshot schemaVersion: "
            f"{schema_version!r}"
        )

    approval_store = ApprovalStateStore(
        agent_descriptors=agent_descriptors,
        graph_runtime=graph_runtime,
        plan_validator=plan_validator,
    )
    approval_records = _required_mapping(snapshot, "approvalRecords")
    approval_store.restore_records(approval_records)

    request_contexts_by_plan_id = {
        plan_id: RequestContext.restore_snapshot(context_snapshot)
        for plan_id, context_snapshot in _required_mapping(
            snapshot,
            "requestContextsByPlanId",
        ).items()
        if isinstance(plan_id, str) and isinstance(context_snapshot, Mapping)
    }

    surface_registry = SurfaceRouteRegistry()
    surface_registry.restore_snapshot(_required_mapping(snapshot, "surfaceRegistry"))

    return RestoredSessionSnapshot(
        approval_store=approval_store,
        request_contexts_by_plan_id=request_contexts_by_plan_id,
        surface_registry=surface_registry,
        artifact_refs=_redacted_json_safe(
            _required_mapping(snapshot, "artifactRefs")
        ),
    )


def _required_mapping(snapshot: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    value = snapshot.get(field_name)
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return value


def _redacted_json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            safe_path_component(str(key)): _redacted_json_safe(child)
            for key, child in value.items()
        }
    if isinstance(value, list | tuple):
        return [_redacted_json_safe(child) for child in value]
    if isinstance(value, set | frozenset):
        return sorted(_redacted_json_safe(child) for child in value)
    if isinstance(value, str):
        return redact_secret_like_values(value)
    return value


__all__ = [
    "RestoredSessionSnapshot",
    "SNAPSHOT_SCHEMA_VERSION",
    "export_session_snapshot",
    "restore_session_snapshot",
]
