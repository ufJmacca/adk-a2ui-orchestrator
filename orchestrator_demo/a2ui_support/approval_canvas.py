"""Editable A2UI approval canvas generation for draft execution plans."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from orchestrator_demo.a2a_support.transport import DataPart
from orchestrator_demo.a2ui_support.schema_manager import (
    A2UI_VERSION,
    BASIC_CATALOG_ID,
)
from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui
from orchestrator_demo.contracts import AgentDescriptor, ExecutionPlan, PlanStep


class A2UIEmissionError(ValueError):
    """Raised when generated A2UI cannot be safely emitted as a DataPart."""


def build_approval_canvas(
    plan: ExecutionPlan,
    *,
    agent_descriptors: Sequence[AgentDescriptor],
) -> list[dict[str, Any]]:
    """Build ordered A2UI Basic Catalog messages for an approval canvas."""

    surface_id = plan.approval_surface_id or f"surface_{plan.plan_id}"
    return [
        {
            "version": A2UI_VERSION,
            "createSurface": {
                "surfaceId": surface_id,
                "catalogId": BASIC_CATALOG_ID,
            },
        },
        {
            "version": A2UI_VERSION,
            "updateComponents": {
                "surfaceId": surface_id,
                "components": _approval_canvas_components(
                    plan,
                    surface_id,
                    agent_descriptors,
                ),
            },
        },
    ]


def approval_canvas_data_parts(
    plan: ExecutionPlan,
    *,
    agent_descriptors: Sequence[AgentDescriptor],
) -> list[DataPart]:
    """Return validated A2UI DataParts for an approval canvas."""

    parts: list[DataPart] = []
    for payload in build_approval_canvas(
        plan,
        agent_descriptors=agent_descriptors,
    ):
        validation_result = validate_outbound_a2ui(payload)
        if not isinstance(validation_result.renderer_part, DataPart):
            errors = "; ".join(validation_result.validation_errors)
            raise A2UIEmissionError(
                "approval canvas failed A2UI validation and was not emitted: "
                f"{errors}"
            )
        parts.append(validation_result.renderer_part)

    return parts


def approval_canvas_data_part(
    plan: ExecutionPlan,
    *,
    agent_descriptors: Sequence[AgentDescriptor],
) -> list[DataPart]:
    """Return validated A2UI DataParts for the approval canvas."""

    return approval_canvas_data_parts(
        plan,
        agent_descriptors=agent_descriptors,
    )


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


def _approval_canvas_components(
    plan: ExecutionPlan,
    surface_id: str,
    agent_descriptors: Sequence[AgentDescriptor],
) -> list[dict[str, Any]]:
    component_prefix = f"component_{plan.plan_id}"
    control_components = _control_components(plan, surface_id)
    control_ids = [
        component["id"]
        for component in control_components
        if component.get("component") in {"Button", "TextField"}
    ]
    content_ids = [
        f"{component_prefix}_title",
        f"{component_prefix}_metadata",
        f"{component_prefix}_objective",
        f"{component_prefix}_agents",
        f"{component_prefix}_steps",
        f"{component_prefix}_dependencies",
        f"{component_prefix}_parallel_groups",
        f"{component_prefix}_available_agents",
        f"{component_prefix}_risk_notes",
        f"{component_prefix}_controls",
    ]

    return [
        {
            "component": "Column",
            "id": "root",
            "children": content_ids,
        },
        {
            "component": "Text",
            "id": f"{component_prefix}_title",
            "text": "Approval plan",
            "variant": "h2",
        },
        {
            "component": "Text",
            "id": f"{component_prefix}_metadata",
            "text": (
                f"surfaceId: {surface_id}\n"
                f"planId: {plan.plan_id}\n"
                f"planVersion: {plan.plan_version}"
            ),
        },
        {
            "component": "Text",
            "id": f"{component_prefix}_objective",
            "text": f"Objective: {plan.objective}",
        },
        {
            "component": "Text",
            "id": f"{component_prefix}_agents",
            "text": _selected_agents_text(plan, agent_descriptors),
        },
        {
            "component": "Text",
            "id": f"{component_prefix}_steps",
            "text": _steps_text(plan.steps),
        },
        {
            "component": "Text",
            "id": f"{component_prefix}_dependencies",
            "text": _dependencies_text(plan.steps),
        },
        {
            "component": "Text",
            "id": f"{component_prefix}_parallel_groups",
            "text": _parallel_groups_text(plan.steps),
        },
        {
            "component": "Text",
            "id": f"{component_prefix}_available_agents",
            "text": _available_agents_text(agent_descriptors),
        },
        {
            "component": "Text",
            "id": f"{component_prefix}_risk_notes",
            "text": _risk_notes_text(plan.risk_notes),
        },
        {
            "component": "Row",
            "id": f"{component_prefix}_controls",
            "children": control_ids,
        },
        *control_components,
    ]


def _selected_agents_text(
    plan: ExecutionPlan,
    descriptors: Sequence[AgentDescriptor],
) -> str:
    agents = _selected_agent_payloads(plan.selected_agents, descriptors)
    formatted_agents = ", ".join(
        f"{agent['agentId']} ({agent['displayName']})" for agent in agents
    )
    return f"Selected agents: {formatted_agents}"


def _steps_text(steps: Sequence[PlanStep]) -> str:
    lines = [
        (
            f"{index}. {step.step_id}: {step.agent_id} - {step.instruction} "
            f"Expected output: {step.expected_output}"
        )
        for index, step in enumerate(steps, start=1)
    ]
    return "Steps:\n" + "\n".join(lines)


def _dependencies_text(steps: Sequence[PlanStep]) -> str:
    lines = [
        f"{step.step_id} dependsOn: {', '.join(step.depends_on) or 'none'}"
        for step in steps
    ]
    return "Dependencies:\n" + "\n".join(lines)


def _parallel_groups_text(steps: Sequence[PlanStep]) -> str:
    groups = _parallel_group_payloads(steps)
    if not groups:
        return "Parallel groups: none"
    lines = [
        f"{group['groupId']}: {', '.join(group['stepIds'])}"
        for group in groups
    ]
    return "Parallel groups:\n" + "\n".join(lines)


def _available_agents_text(descriptors: Sequence[AgentDescriptor]) -> str:
    agents = _available_agent_payloads(descriptors)
    formatted_agents = ", ".join(
        f"{agent['agentId']} ({agent['displayName']})" for agent in agents
    )
    return f"Available agents: {formatted_agents}"


def _risk_notes_text(risk_notes: Sequence[str]) -> str:
    if not risk_notes:
        return "Risk notes: none"
    return "Risk notes:\n" + "\n".join(risk_notes)


def _control_components(plan: ExecutionPlan, surface_id: str) -> list[dict[str, Any]]:
    step_ids = [step.step_id for step in plan.steps]
    common_payload = {
        "planId": plan.plan_id,
        "planVersion": plan.plan_version,
        "editedPlanVersion": plan.plan_version,
    }
    controls: list[list[dict[str, Any]]] = [
        _button_control(
            "approve_plan",
            "Approve",
            plan,
            surface_id,
            {
                **common_payload,
                "approvedStepIds": step_ids,
            },
            variant="primary",
        ),
        _button_control(
            "reject_plan",
            "Reject",
            plan,
            surface_id,
            {
                **common_payload,
                "reason": "",
            },
        ),
        _button_control(
            "edit_plan",
            "Edit Plan",
            plan,
            surface_id,
            {
                **common_payload,
                "editableFields": ["steps", "selectedAgents"],
            },
        ),
        _button_control(
            "reorder_steps",
            "Reorder Steps",
            plan,
            surface_id,
            {
                **common_payload,
                "orderedStepIds": step_ids,
            },
        ),
    ]
    for step in plan.steps:
        controls.extend(
            _step_edit_controls(
                plan,
                surface_id,
                step,
                common_payload,
            )
        )
    return [
        component
        for control_components in controls
        for component in control_components
    ]


def _step_edit_controls(
    plan: ExecutionPlan,
    surface_id: str,
    step: PlanStep,
    common_payload: dict[str, Any],
) -> list[list[dict[str, Any]]]:
    replacement_agent_path = _data_model_path(
        step,
        "replacementAgentId",
    )
    instruction_path = _data_model_path(step, "instruction")
    return [
        _button_control(
            "remove_step",
            f"Remove {step.step_id}",
            plan,
            surface_id,
            {
                **common_payload,
                "stepId": step.step_id,
            },
            control_id=f"control_remove_step_{step.step_id}",
        ),
        [
            _text_field_control(
                f"control_replace_agent_{step.step_id}_input",
                f"Replacement agent for {step.step_id}",
                replacement_agent_path,
            )
        ],
        _button_control(
            "replace_agent",
            f"Replace agent for {step.step_id}",
            plan,
            surface_id,
            {
                **common_payload,
                "stepId": step.step_id,
                "replacementAgentId": {"path": replacement_agent_path},
            },
            control_id=f"control_replace_agent_{step.step_id}",
        ),
        [
            _text_field_control(
                f"control_add_instruction_{step.step_id}_input",
                f"Instruction for {step.step_id}",
                instruction_path,
                variant="longText",
            )
        ],
        _button_control(
            "add_instruction",
            f"Add instruction to {step.step_id}",
            plan,
            surface_id,
            {
                **common_payload,
                "stepId": step.step_id,
                "instruction": {"path": instruction_path},
            },
            control_id=f"control_add_instruction_{step.step_id}",
        ),
    ]


def _text_field_control(
    control_id: str,
    label: str,
    value_path: str,
    *,
    variant: str = "shortText",
) -> dict[str, Any]:
    return {
        "component": "TextField",
        "id": control_id,
        "label": label,
        "value": {"path": value_path},
        "variant": variant,
    }


def _data_model_path(step: PlanStep, field_name: str) -> str:
    return f"/approvalEdits/{step.step_id}/{field_name}"


def _button_control(
    action_type: str,
    label: str,
    plan: ExecutionPlan,
    surface_id: str,
    payload: dict[str, Any],
    *,
    variant: str = "default",
    control_id: str | None = None,
) -> list[dict[str, Any]]:
    button_id = control_id or f"control_{action_type}"
    label_id = f"{button_id}_label"
    context = {
        "type": action_type,
        "surfaceId": surface_id,
        "payload": payload,
    }
    return [
        {
            "component": "Button",
            "id": button_id,
            "child": label_id,
            "variant": variant,
            "action": {
                "event": {
                    "name": action_type,
                    "context": context,
                }
            },
        },
        {
            "component": "Text",
            "id": label_id,
            "text": label,
        },
    ]


def _display_name_from_agent_id(agent_id: str) -> str:
    return f"{agent_id.replace('_', ' ').title()} Agent"


__all__ = [
    "A2UIEmissionError",
    "approval_canvas_data_part",
    "approval_canvas_data_parts",
    "build_approval_canvas",
]
