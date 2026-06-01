"""Minimal Basic Catalog schema checks for outbound A2UI payloads."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


BASIC_CATALOG_NAME = "basic"
WORKFLOW_CANVAS_TYPE = "workflowCanvas"
ALLOWED_CONTROL_ACTIONS = {
    "approve_plan",
    "reject_plan",
    "edit_plan",
    "remove_step",
    "reorder_steps",
    "replace_agent",
    "choose_agent",
    "add_instruction",
    "add_instructions",
}
SURFACE_ID_PATTERN = re.compile(r"^surface_[A-Za-z0-9][A-Za-z0-9_-]*$")
PLAN_ID_PATTERN = re.compile(r"^plan_[A-Za-z0-9][A-Za-z0-9_-]*$")
STEP_ID_PATTERN = re.compile(r"^step_[A-Za-z0-9][A-Za-z0-9_-]*$")


class BasicCatalogSchema:
    """Validate the Basic Catalog subset used by the demo renderer."""

    def validate(self, payload: Mapping[str, Any]) -> list[str]:
        errors: list[str] = []

        _require_string(
            payload,
            "catalog",
            errors,
            expected_value=BASIC_CATALOG_NAME,
        )
        _require_pattern(payload, "surfaceId", SURFACE_ID_PATTERN, errors)
        _require_pattern(payload, "planId", PLAN_ID_PATTERN, errors)
        _require_positive_int(payload, "planVersion", errors)
        _require_string(
            payload,
            "kind",
            errors,
            expected_value=WORKFLOW_CANVAS_TYPE,
        )

        components = payload.get("components")
        if not isinstance(components, list) or not components:
            errors.append("components must be a non-empty list")
            return errors

        for index, component in enumerate(components):
            path = f"components[{index}]"
            if not isinstance(component, Mapping):
                errors.append(f"{path} must be an object")
                continue
            component_type = component.get("type")
            if component_type == WORKFLOW_CANVAS_TYPE:
                _validate_workflow_canvas_component(component, path, payload, errors)
            elif not isinstance(component_type, str) or not component_type:
                errors.append(f"{path}.type must be a non-empty string")

        return errors


def validate_basic_catalog_payload(payload: Mapping[str, Any]) -> list[str]:
    """Return schema validation errors for a Basic Catalog payload."""

    return BasicCatalogSchema().validate(payload)


def _validate_workflow_canvas_component(
    component: Mapping[str, Any],
    path: str,
    root_payload: Mapping[str, Any],
    errors: list[str],
) -> None:
    _require_string(component, "id", errors, path=path)
    _require_string(component, "objective", errors, path=path)
    _validate_selected_agents(
        component.get("selectedAgents"),
        f"{path}.selectedAgents",
        errors,
    )
    _validate_steps(component.get("steps"), f"{path}.steps", errors)
    _validate_parallel_groups(
        component.get("parallelGroups"),
        f"{path}.parallelGroups",
        component.get("steps"),
        errors,
    )
    _validate_controls(
        component.get("controls"),
        f"{path}.controls",
        root_payload,
        errors,
    )


def _validate_selected_agents(value: Any, path: str, errors: list[str]) -> None:
    if not isinstance(value, list) or not value:
        errors.append(f"{path} must be a non-empty list")
        return

    for index, agent in enumerate(value):
        agent_path = f"{path}[{index}]"
        if not isinstance(agent, Mapping):
            errors.append(f"{agent_path} must be an object")
            continue
        _require_string(agent, "agentId", errors, path=agent_path)
        _require_string(agent, "displayName", errors, path=agent_path)


def _validate_steps(value: Any, path: str, errors: list[str]) -> None:
    if not isinstance(value, list) or not value:
        errors.append(f"{path} must be a non-empty list")
        return

    declared_step_ids: set[str] = set()
    for index, step in enumerate(value):
        step_path = f"{path}[{index}]"
        if not isinstance(step, Mapping):
            errors.append(f"{step_path} must be an object")
            continue

        step_id = step.get("stepId")
        if not isinstance(step_id, str) or not STEP_ID_PATTERN.match(step_id):
            errors.append(f"{step_path}.stepId must match step id format")
        elif step_id in declared_step_ids:
            errors.append(f"{step_path}.stepId must be unique")
        else:
            declared_step_ids.add(step_id)

        _require_string(step, "agentId", errors, path=step_path)
        _require_string(step, "instruction", errors, path=step_path)
        _require_string(step, "expectedOutput", errors, path=step_path)

        depends_on = step.get("dependsOn")
        if not isinstance(depends_on, list):
            errors.append(f"{step_path}.dependsOn must be a list")
        elif any(not isinstance(dependency, str) for dependency in depends_on):
            errors.append(f"{step_path}.dependsOn values must be strings")

        parallel_group = step.get("parallelGroup")
        if parallel_group is not None and not isinstance(parallel_group, str):
            errors.append(f"{step_path}.parallelGroup must be a string or null")

    for index, step in enumerate(value):
        if not isinstance(step, Mapping) or not isinstance(step.get("dependsOn"), list):
            continue
        missing_dependencies = [
            dependency
            for dependency in step["dependsOn"]
            if dependency not in declared_step_ids
        ]
        if missing_dependencies:
            errors.append(
                f"{path}[{index}].dependsOn references unknown steps: "
                f"{', '.join(missing_dependencies)}"
            )


def _validate_parallel_groups(
    value: Any,
    path: str,
    steps: Any,
    errors: list[str],
) -> None:
    if not isinstance(value, list):
        errors.append(f"{path} must be a list")
        return

    declared_step_ids: set[str] = set()
    if isinstance(steps, list):
        declared_step_ids = {
            step["stepId"]
            for step in steps
            if isinstance(step, Mapping)
            if isinstance(step.get("stepId"), str)
        }
    for index, group in enumerate(value):
        group_path = f"{path}[{index}]"
        if not isinstance(group, Mapping):
            errors.append(f"{group_path} must be an object")
            continue
        _require_string(group, "groupId", errors, path=group_path)
        step_ids = group.get("stepIds")
        if not isinstance(step_ids, list) or not step_ids:
            errors.append(f"{group_path}.stepIds must be a non-empty list")
            continue
        unknown_step_ids = [
            step_id
            for step_id in step_ids
            if not isinstance(step_id, str) or step_id not in declared_step_ids
        ]
        if unknown_step_ids:
            errors.append(
                f"{group_path}.stepIds references unknown steps: "
                f"{', '.join(str(step_id) for step_id in unknown_step_ids)}"
            )


def _validate_controls(
    value: Any,
    path: str,
    root_payload: Mapping[str, Any],
    errors: list[str],
) -> None:
    if not isinstance(value, list) or not value:
        errors.append(f"{path} must be a non-empty list")
        return

    for index, control in enumerate(value):
        control_path = f"{path}[{index}]"
        if not isinstance(control, Mapping):
            errors.append(f"{control_path} must be an object")
            continue

        _require_string(control, "controlId", errors, path=control_path)
        _require_string(control, "type", errors, path=control_path)
        _require_string(control, "label", errors, path=control_path)

        action = control.get("action")
        action_path = f"{control_path}.action"
        if not isinstance(action, Mapping):
            errors.append(f"{action_path} must be an object")
            continue

        action_type = action.get("type")
        if action_type not in ALLOWED_CONTROL_ACTIONS:
            errors.append(f"{action_path}.type must be a supported plan action")

        for field_name in ("surfaceId", "planId", "planVersion"):
            if action.get(field_name) != root_payload.get(field_name):
                errors.append(
                    f"{action_path}.{field_name} must match payload {field_name}"
                )

        payload = action.get("payload")
        if not isinstance(payload, Mapping):
            errors.append(f"{action_path}.payload must be an object")


def _require_string(
    payload: Mapping[str, Any],
    field_name: str,
    errors: list[str],
    *,
    path: str | None = None,
    expected_value: str | None = None,
) -> None:
    value = payload.get(field_name)
    field_path = f"{path}.{field_name}" if path else field_name
    if not isinstance(value, str) or not value:
        errors.append(f"{field_path} must be a non-empty string")
        return
    if expected_value is not None and value != expected_value:
        errors.append(f"{field_path} must be {expected_value!r}")


def _require_pattern(
    payload: Mapping[str, Any],
    field_name: str,
    pattern: re.Pattern[str],
    errors: list[str],
) -> None:
    value = payload.get(field_name)
    if not isinstance(value, str) or not pattern.match(value):
        errors.append(f"{field_name} must match {pattern.pattern}")


def _require_positive_int(
    payload: Mapping[str, Any],
    field_name: str,
    errors: list[str],
) -> None:
    value = payload.get(field_name)
    if not isinstance(value, int) or value < 1:
        errors.append(f"{field_name} must be a positive integer")


__all__ = [
    "ALLOWED_CONTROL_ACTIONS",
    "BASIC_CATALOG_NAME",
    "BasicCatalogSchema",
    "WORKFLOW_CANVAS_TYPE",
    "validate_basic_catalog_payload",
]
