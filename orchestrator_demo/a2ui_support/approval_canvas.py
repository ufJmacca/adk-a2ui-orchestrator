"""Editable A2UI approval canvas generation for draft execution plans."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from orchestrator_demo.a2a_support.transport import DataPart
from orchestrator_demo.a2ui_support.schema_manager import (
    BASIC_CATALOG_NAME,
    WORKFLOW_CANVAS_TYPE,
)
from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui
from orchestrator_demo.contracts import AgentDescriptor, ExecutionPlan, PlanStep


class A2UIEmissionError(ValueError):
    """Raised when generated A2UI cannot be safely emitted as a DataPart."""


def build_approval_canvas(
    plan: ExecutionPlan,
    *,
    agent_descriptors: Sequence[AgentDescriptor],
) -> dict[str, Any]:
    """Build Basic Catalog-compatible JSON for editable plan approval."""

    surface_id = plan.approval_surface_id or f"surface_{plan.plan_id}"
    payload = {
        "catalog": BASIC_CATALOG_NAME,
        "surfaceId": surface_id,
        "planId": plan.plan_id,
        "planVersion": plan.plan_version,
        "kind": WORKFLOW_CANVAS_TYPE,
        "components": [
            {
                "type": WORKFLOW_CANVAS_TYPE,
                "id": f"component_{plan.plan_id}_canvas",
                "objective": plan.objective,
                "detectedIntents": list(plan.detected_intents),
                "selectedAgents": _selected_agent_payloads(
                    plan.selected_agents,
                    agent_descriptors,
                ),
                "steps": [_step_payload(step) for step in plan.steps],
                "parallelGroups": _parallel_group_payloads(plan.steps),
                "dataSourceCategories": list(plan.data_source_categories),
                "riskNotes": list(plan.risk_notes),
                "availableAgents": _available_agent_payloads(agent_descriptors),
                "controls": _control_payloads(plan, surface_id),
            }
        ],
    }
    return payload


def approval_canvas_data_part(
    plan: ExecutionPlan,
    *,
    agent_descriptors: Sequence[AgentDescriptor],
) -> DataPart:
    """Return a validated A2UI DataPart for an approval canvas."""

    payload = build_approval_canvas(plan, agent_descriptors=agent_descriptors)
    validation_result = validate_outbound_a2ui(payload)
    if not isinstance(validation_result.renderer_part, DataPart):
        errors = "; ".join(validation_result.validation_errors)
        raise A2UIEmissionError(
            f"approval canvas failed A2UI validation and was not emitted: {errors}"
        )

    return validation_result.renderer_part


def _selected_agent_payloads(
    selected_agent_ids: Sequence[str],
    descriptors: Sequence[AgentDescriptor],
) -> list[dict[str, Any]]:
    descriptors_by_id = {descriptor.agent_id: descriptor for descriptor in descriptors}
    return [
        _agent_payload(
            agent_id,
            descriptors_by_id.get(agent_id),
        )
        for agent_id in selected_agent_ids
    ]


def _available_agent_payloads(
    descriptors: Sequence[AgentDescriptor],
) -> list[dict[str, Any]]:
    return [
        {
            "agentId": descriptor.agent_id,
            "displayName": descriptor.display_name,
            "capabilities": list(descriptor.capabilities),
        }
        for descriptor in descriptors
    ]


def _agent_payload(
    agent_id: str,
    descriptor: AgentDescriptor | None,
) -> dict[str, Any]:
    if descriptor is None:
        return {
            "agentId": agent_id,
            "displayName": _display_name_from_agent_id(agent_id),
        }

    return {
        "agentId": descriptor.agent_id,
        "displayName": descriptor.display_name,
    }


def _step_payload(step: PlanStep) -> dict[str, Any]:
    return {
        "stepId": step.step_id,
        "agentId": step.agent_id,
        "instruction": step.instruction,
        "dependsOn": list(step.depends_on),
        "expectedOutput": step.expected_output,
        "dataSourceCategories": list(step.data_source_categories),
        "parallelGroup": step.parallel_group,
    }


def _parallel_group_payloads(steps: Sequence[PlanStep]) -> list[dict[str, Any]]:
    step_ids_by_group: dict[str, list[str]] = {}
    for step in steps:
        if step.parallel_group is None:
            continue
        step_ids_by_group.setdefault(step.parallel_group, []).append(step.step_id)

    return [
        {"groupId": group_id, "stepIds": step_ids}
        for group_id, step_ids in step_ids_by_group.items()
    ]


def _control_payloads(plan: ExecutionPlan, surface_id: str) -> list[dict[str, Any]]:
    step_ids = [step.step_id for step in plan.steps]
    common_payload = {
        "planId": plan.plan_id,
        "editedPlanVersion": plan.plan_version,
    }
    return [
        _control(
            "approve_plan",
            "Approve",
            plan,
            surface_id,
            {
                **common_payload,
                "approvedStepIds": step_ids,
            },
        ),
        _control(
            "reject_plan",
            "Reject",
            plan,
            surface_id,
            {
                "planId": plan.plan_id,
                "reason": "",
            },
        ),
        _control(
            "edit_plan",
            "Edit Plan",
            plan,
            surface_id,
            {
                **common_payload,
                "editableFields": ["objective", "steps", "selectedAgents"],
            },
        ),
        _control(
            "remove_step",
            "Remove Step",
            plan,
            surface_id,
            {
                **common_payload,
                "stepId": None,
            },
        ),
        _control(
            "reorder_steps",
            "Reorder Steps",
            plan,
            surface_id,
            {
                **common_payload,
                "orderedStepIds": step_ids,
            },
        ),
        _control(
            "replace_agent",
            "Replace Agent",
            plan,
            surface_id,
            {
                **common_payload,
                "stepId": None,
                "replacementAgentId": None,
            },
        ),
        _control(
            "add_instruction",
            "Add Instruction",
            plan,
            surface_id,
            {
                **common_payload,
                "stepId": None,
                "instruction": "",
            },
        ),
    ]


def _control(
    action_type: str,
    label: str,
    plan: ExecutionPlan,
    surface_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "controlId": f"control_{action_type}",
        "type": "button",
        "label": label,
        "action": {
            "type": action_type,
            "surfaceId": surface_id,
            "planId": plan.plan_id,
            "planVersion": plan.plan_version,
            "payload": payload,
        },
    }


def _display_name_from_agent_id(agent_id: str) -> str:
    return f"{agent_id.replace('_', ' ').title()} Agent"


__all__ = [
    "A2UIEmissionError",
    "approval_canvas_data_part",
    "build_approval_canvas",
]
