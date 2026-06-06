"""Deterministic ADK tool response payload builders."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from typing import Any

from pydantic import BaseModel

from orchestrator_demo.a2a_support.transport import DataPart
from orchestrator_demo.a2ui_support.approval_canvas import A2UIEmissionError
from orchestrator_demo.a2ui_support.renderer_contract import RendererContractError
from orchestrator_demo.a2ui_support.secret_safety import redact_secret_like_values
from orchestrator_demo.contracts import ExecutionPlan, PlanStep
from orchestrator_demo.orchestrator.approval_state import (
    ApprovalActionResult,
    ApprovalStateError,
    PlanAlreadyFinalError,
    PlanMutationError,
    PlanNotFoundError,
    PlanSurfaceMismatchError,
    PlanVersionConflictError,
)
from orchestrator_demo.orchestrator.graph_runtime import (
    AdkGraphApiError,
    GraphRuntimeError,
)
from orchestrator_demo.orchestrator.service import (
    OrchestratorRequestResult,
    OrchestratorUserActionResult,
)
from orchestrator_demo.orchestrator.surface_routes import SurfaceOwnershipError


class ArtifactStorageError(RuntimeError):
    """Raised when ADK artifact persistence fails."""


def build_request_response(result: OrchestratorRequestResult) -> dict[str, Any]:
    """Build a JSON-serializable ADK tool response for a new user request."""

    payload: dict[str, Any] = {
        "status": result.path,
        "path": result.path,
        "decision": _jsonable_camel(result.decision),
        "approvalPlan": _jsonable_legacy(result.approval_plan),
        "approvalResult": _approval_result_payload(result.approval_result),
        "specialistResponses": _jsonable_legacy(result.specialist_responses),
        "a2uiParts": _data_parts_payload(result.a2ui_parts),
        "nextActions": [],
        "statusEvents": _jsonable_legacy(result.status_events),
        "artifacts": _jsonable_legacy(result.final_artifacts),
    }
    if result.approval_plan is not None:
        payload.update(_plan_state_fields(result.approval_plan, editable=True))
    return payload


def build_user_action_response(
    result: OrchestratorUserActionResult,
) -> dict[str, Any]:
    """Build a JSON-serializable ADK tool response for a structured action."""

    if result.status == "error" and result.surface_route_result is not None:
        return _surface_route_error_payload(result)

    approval_result = result.approval_result
    payload: dict[str, Any] = {
        "status": result.status,
        "path": result.status,
        "approvalResult": _approval_result_payload(approval_result),
        "surfaceRouteResult": _surface_route_result_payload(
            result.surface_route_result
        ),
        "specialistResponses": _jsonable_legacy(result.specialist_responses),
        "a2uiParts": _data_parts_payload(result.a2ui_parts),
        "nextActions": [],
        "statusEvents": _jsonable_legacy(result.status_events),
        "artifacts": _jsonable_legacy(result.final_artifacts),
    }

    plan = _result_plan_for_contract(approval_result)
    if plan is not None:
        payload.update(
            _plan_state_fields(plan, editable=result.status == "draft_updated")
        )
    elif approval_result is not None:
        if approval_result.plan_id is not None:
            payload["planId"] = approval_result.plan_id
        if approval_result.plan_version is not None:
            payload["planVersion"] = approval_result.plan_version

    return payload


def build_error_response(exc: BaseException) -> dict[str, Any]:
    """Map a known exception to a safe, stable ADK tool error response."""

    code, message = _error_code_and_message(exc)
    error: dict[str, Any] = {
        "code": code,
        "message": message,
    }
    detail = _safe_error_detail(exc)
    if detail:
        error["detail"] = detail

    return {
        "status": "error",
        "path": "error",
        "error": error,
        "a2uiParts": [],
        "nextActions": [],
        "statusEvents": [],
        "artifacts": {},
    }


def _surface_route_error_payload(
    result: OrchestratorUserActionResult,
) -> dict[str, Any]:
    route_result = result.surface_route_result
    error = dict(route_result.error or {}) if route_result is not None else {}
    safe_error = {
        "code": "surface_route_error",
        "message": "The A2UI surface action could not be routed.",
    }
    if error.get("surfaceId") is not None:
        safe_error["surfaceId"] = _safe_text(str(error["surfaceId"]))
    route_code = _safe_surface_route_error_code(error.get("code"))
    if route_code != safe_error["code"]:
        safe_error["routeCode"] = route_code

    return {
        "status": "error",
        "path": "error",
        "error": safe_error,
        "surfaceRouteResult": _surface_route_result_payload(route_result),
        "a2uiParts": _data_parts_payload(result.a2ui_parts),
        "nextActions": [],
        "statusEvents": _jsonable_legacy(result.status_events),
        "artifacts": _jsonable_legacy(result.final_artifacts),
    }


def _surface_route_result_payload(
    route_result: Any,
) -> dict[str, Any] | None:
    if route_result is None:
        return None

    owner = route_result.owner
    return {
        "status": route_result.status,
        "surfaceId": _safe_text(str(route_result.surface_id))
        if route_result.surface_id is not None
        else None,
        "owner": None
        if owner is None
        else {
            "surfaceId": _safe_text(owner.surface_id),
            "ownerType": owner.owner_type,
            "ownerId": _safe_text(owner.owner_id),
            "planId": _safe_text(owner.plan_id) if owner.plan_id is not None else None,
        },
        "error": _surface_route_result_error_payload(route_result.error),
    }


def _surface_route_result_error_payload(
    error: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if error is None:
        return None

    payload: dict[str, Any] = {
        "code": _safe_surface_route_error_code(error.get("code")),
    }
    if error.get("surfaceId") is not None:
        payload["surfaceId"] = _safe_text(str(error["surfaceId"]))
    if error.get("ownerInferenceAttempted") is not None:
        payload["ownerInferenceAttempted"] = bool(error["ownerInferenceAttempted"])
    return payload


def _safe_surface_route_error_code(value: Any) -> str:
    code = str(value) if value is not None else ""
    if code in {
        "invalid_user_action",
        "owner_handler_failed",
        "owner_unavailable",
        "surface_registration_rejected",
        "unknown_surface",
    }:
        return code
    return "surface_route_error"


def _approval_result_payload(
    result: ApprovalActionResult | None,
) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "status": result.status,
        "planId": result.plan_id,
        "planVersion": result.plan_version,
        "draftPlan": _plan_payload(result.draft_plan),
        "approvedPlan": _plan_payload(result.approved_plan),
        "reason": _safe_text(result.rejection_reason)
        if result.rejection_reason is not None
        else None,
        "graphCreated": result.graph_created,
        "specialistsCalled": result.specialists_called,
    }


def _result_plan_for_contract(
    approval_result: ApprovalActionResult | None,
) -> ExecutionPlan | None:
    if approval_result is None:
        return None
    if approval_result.status == "draft_updated":
        return approval_result.draft_plan
    if approval_result.status == "approved":
        return approval_result.approved_plan
    return None


def _plan_state_fields(plan: ExecutionPlan, *, editable: bool) -> dict[str, Any]:
    return {
        "planId": plan.plan_id,
        "planVersion": plan.plan_version,
        "approvalSurfaceId": _approval_surface_id(plan),
        "selectedAgents": list(plan.selected_agents),
        "stepIds": [step.step_id for step in plan.steps],
        "stepInstructions": [
            {"stepId": step.step_id, "instruction": step.instruction}
            for step in plan.steps
        ],
        "dependencies": [
            {"stepId": step.step_id, "dependsOn": list(step.depends_on)}
            for step in plan.steps
        ],
        "riskNotes": list(plan.risk_notes),
        "plan": _plan_payload(plan),
        "nextActions": _next_actions(plan) if editable else [],
    }


def _plan_payload(plan: ExecutionPlan | None) -> dict[str, Any] | None:
    if plan is None:
        return None
    return {
        "planId": plan.plan_id,
        "objective": plan.objective,
        "detectedIntents": list(plan.detected_intents),
        "selectedAgents": list(plan.selected_agents),
        "steps": [_step_payload(step) for step in plan.steps],
        "dataSourceCategories": list(plan.data_source_categories),
        "riskNotes": list(plan.risk_notes),
        "approvalSurfaceId": _approval_surface_id(plan),
        "planVersion": plan.plan_version,
        "immutableAfterApproval": plan.immutable_after_approval,
    }


def _step_payload(step: PlanStep) -> dict[str, Any]:
    return {
        "stepId": step.step_id,
        "agentId": step.agent_id,
        "instruction": step.instruction,
        "dependsOn": list(step.depends_on),
        "condition": step.condition,
        "expectedOutput": step.expected_output,
        "dataSourceCategories": list(step.data_source_categories),
        "parallelGroup": step.parallel_group,
    }


def _next_actions(plan: ExecutionPlan) -> list[dict[str, Any]]:
    common_context = {
        "planId": plan.plan_id,
        "approvalSurfaceId": _approval_surface_id(plan),
        "editedPlanVersion": plan.plan_version,
    }
    step_ids = [step.step_id for step in plan.steps]
    return [
        {
            "toolName": "add_plan_instruction",
            **common_context,
            "requiredFields": [
                "plan_id",
                "approval_surface_id",
                "edited_plan_version",
                "step_id",
                "instruction",
            ],
        },
        {
            "toolName": "remove_plan_step",
            **common_context,
            "requiredFields": [
                "plan_id",
                "approval_surface_id",
                "edited_plan_version",
                "step_id",
            ],
        },
        {
            "toolName": "replace_plan_agent",
            **common_context,
            "requiredFields": [
                "plan_id",
                "approval_surface_id",
                "edited_plan_version",
                "step_id",
                "replacement_agent_id",
            ],
        },
        {
            "toolName": "reorder_plan_steps",
            **common_context,
            "requiredFields": [
                "plan_id",
                "approval_surface_id",
                "edited_plan_version",
                "ordered_step_ids",
            ],
        },
        {
            "toolName": "approve_orchestrator_plan",
            **common_context,
            "approvedStepIds": step_ids,
            "requiredFields": [
                "plan_id",
                "approval_surface_id",
                "edited_plan_version",
                "approved_step_ids",
            ],
        },
        {
            "toolName": "reject_orchestrator_plan",
            **common_context,
            "requiredFields": [
                "plan_id",
                "approval_surface_id",
                "reason",
            ],
        },
    ]


def _approval_surface_id(plan: ExecutionPlan) -> str:
    return plan.approval_surface_id or f"surface_{plan.plan_id}"


def _data_parts_payload(parts: Sequence[DataPart]) -> list[dict[str, Any]]:
    return [part.model_dump(by_alias=True, mode="json") for part in parts]


def _jsonable_camel(value: Any) -> Any:
    return _jsonable(value, camel_case_keys=True)


def _jsonable_legacy(value: Any) -> Any:
    return _jsonable(value, camel_case_keys=False)


def _jsonable(value: Any, *, camel_case_keys: bool) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, BaseModel):
        return _jsonable(value.model_dump(mode="json"), camel_case_keys=camel_case_keys)
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value), camel_case_keys=camel_case_keys)
    if isinstance(value, Mapping):
        return {
            _maybe_camel_case(str(key), enabled=camel_case_keys): _jsonable(
                child,
                camel_case_keys=camel_case_keys,
            )
            for key, child in value.items()
        }
    if isinstance(value, tuple | list):
        return [_jsonable(child, camel_case_keys=camel_case_keys) for child in value]
    return str(value)


def _maybe_camel_case(value: str, *, enabled: bool) -> str:
    if not enabled:
        return value
    parts = value.split("_")
    if len(parts) == 1:
        return value
    return parts[0] + "".join(part[:1].upper() + part[1:] for part in parts[1:])


def _error_code_and_message(exc: BaseException) -> tuple[str, str]:
    if isinstance(exc, PlanNotFoundError):
        return "plan_not_found", "The requested plan was not found."
    if isinstance(exc, PlanVersionConflictError):
        return "stale_plan_version", "The plan version is stale."
    if isinstance(exc, PlanSurfaceMismatchError):
        return "surface_mismatch", "The approval surface does not match the plan."
    if isinstance(exc, PlanAlreadyFinalError):
        return "plan_already_final", "The plan is already final."
    if isinstance(exc, PlanMutationError):
        return "invalid_plan_mutation", "The plan mutation was rejected."
    if isinstance(exc, GraphRuntimeError | AdkGraphApiError):
        return "graph_execution_failed", "The approved graph could not be executed."
    if isinstance(exc, SurfaceOwnershipError):
        return "surface_ownership_error", "The A2UI surface ownership change failed."
    if isinstance(exc, A2UIEmissionError | RendererContractError):
        return "a2ui_delivery_error", "The A2UI payload could not be delivered safely."
    if isinstance(exc, ArtifactStorageError | OSError):
        return "artifact_storage_error", "The artifact could not be saved."
    if isinstance(exc, ApprovalStateError):
        return "approval_state_error", "The approval state transition was rejected."
    return "unexpected_error", "The orchestrator operation failed."


def _safe_error_detail(exc: BaseException) -> str | None:
    safe_lines: list[str] = []
    for line in _safe_text(str(exc)).splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if lowered.startswith("traceback") or stripped.startswith("File "):
            continue
        safe_lines.append(stripped)

    if not safe_lines:
        return None
    return safe_lines[0][:300]


def _safe_text(value: str) -> str:
    return redact_secret_like_values(value)


__all__ = [
    "ArtifactStorageError",
    "build_error_response",
    "build_request_response",
    "build_user_action_response",
]
