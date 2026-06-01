"""Minimal Basic Catalog schema checks for outbound A2UI payloads."""

from __future__ import annotations

import re
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

from a2ui.basic_catalog.provider import (  # type: ignore[import-untyped]
    BasicCatalog as SdkBasicCatalog,
)
from a2ui.schema.constants import VERSION_0_9  # type: ignore[import-untyped]
from a2ui.schema.manager import A2uiSchemaManager  # type: ignore[import-untyped]


BASIC_CATALOG_NAME = "basic"
BASIC_CATALOG_ID = "https://a2ui.org/specification/v0_9/basic_catalog.json"
A2UI_VERSION = "v0.9"
WORKFLOW_CANVAS_TYPE = "workflowCanvas"
PLAN_APPROVAL_SURFACE_PREFIX = "surface_plan_"
CREATE_SURFACE_MESSAGE = "createSurface"
UPDATE_COMPONENTS_MESSAGE = "updateComponents"
UPDATE_DATA_MODEL_MESSAGE = "updateDataModel"
DELETE_SURFACE_MESSAGE = "deleteSurface"
A2UI_SERVER_TO_CLIENT_MESSAGES = {
    CREATE_SURFACE_MESSAGE,
    UPDATE_COMPONENTS_MESSAGE,
    UPDATE_DATA_MODEL_MESSAGE,
    DELETE_SURFACE_MESSAGE,
}
SUPPORTED_BASIC_COMPONENT_TYPES = {
    "AudioPlayer",
    "Button",
    "Card",
    "CheckBox",
    "ChoicePicker",
    "Column",
    "DateTimeInput",
    "Divider",
    "Icon",
    "Image",
    "List",
    "Modal",
    "Row",
    "Slider",
    "Tabs",
    "Text",
    "TextField",
    "Video",
    "audioPlayer",
    "button",
    "card",
    "checkBox",
    "choicePicker",
    "column",
    "dateTimeInput",
    "divider",
    "icon",
    "image",
    "list",
    "modal",
    "row",
    "slider",
    "tabs",
    "text",
    "textField",
    "video",
}
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
    """Validate Basic Catalog A2UI server-to-client envelopes."""

    def validate(self, payload: Mapping[str, Any]) -> list[str]:
        errors: list[str] = []

        message_type = _validate_server_to_client_envelope(payload, errors)
        if message_type is None:
            return errors

        _validate_with_a2ui_sdk(payload, errors)

        if message_type == CREATE_SURFACE_MESSAGE:
            _validate_create_surface(payload[CREATE_SURFACE_MESSAGE], errors)
            return errors

        if message_type != UPDATE_COMPONENTS_MESSAGE:
            _validate_surface_message(payload[message_type], message_type, errors)
            return errors

        update_components = payload[UPDATE_COMPONENTS_MESSAGE]
        if not isinstance(update_components, Mapping):
            errors.append(f"{UPDATE_COMPONENTS_MESSAGE} must be an object")
            return errors

        _require_pattern(
            update_components,
            "surfaceId",
            SURFACE_ID_PATTERN,
            errors,
            path=UPDATE_COMPONENTS_MESSAGE,
        )

        components = update_components.get("components")
        if not isinstance(components, list) or not components:
            errors.append(
                f"{UPDATE_COMPONENTS_MESSAGE}.components must be a non-empty list"
            )
            return errors

        _validate_generic_components(
            components,
            errors,
            path=f"{UPDATE_COMPONENTS_MESSAGE}.components",
        )
        if _is_approval_canvas_payload(payload, components):
            _validate_component_graph(
                components,
                errors,
                path=f"{UPDATE_COMPONENTS_MESSAGE}.components",
            )
        return errors


def _is_approval_canvas_payload(
    payload: Mapping[str, Any],
    components: list[Any],
) -> bool:
    if payload.get("kind") == WORKFLOW_CANVAS_TYPE:
        return True
    update_components = payload.get(UPDATE_COMPONENTS_MESSAGE)
    if isinstance(update_components, Mapping):
        surface_id = update_components.get("surfaceId")
        if (
            isinstance(surface_id, str)
            and surface_id.startswith(PLAN_APPROVAL_SURFACE_PREFIX)
        ):
            return True
    return any(
        _is_workflow_canvas_component(component)
        for component in components
    )


def _is_workflow_canvas_component(component: Any) -> bool:
    return (
        isinstance(component, Mapping)
        and (
            component.get("type") == WORKFLOW_CANVAS_TYPE
            or component.get("component") == WORKFLOW_CANVAS_TYPE
        )
    )


def _validate_server_to_client_envelope(
    payload: Mapping[str, Any],
    errors: list[str],
) -> str | None:
    version = payload.get("version")
    if version != A2UI_VERSION:
        errors.append(f"version must be {A2UI_VERSION!r}")

    message_types = [
        message_type
        for message_type in A2UI_SERVER_TO_CLIENT_MESSAGES
        if message_type in payload
    ]
    if len(message_types) != 1:
        errors.append(
            "A2UI payload must contain exactly one server-to-client message: "
            f"{', '.join(sorted(A2UI_SERVER_TO_CLIENT_MESSAGES))}"
        )
        return None

    allowed_keys = {"version", message_types[0]}
    extra_keys = sorted(set(payload) - allowed_keys)
    if extra_keys:
        errors.append(
            "A2UI payload has unsupported top-level keys "
            f"({len(extra_keys)} present)"
        )

    message = payload.get(message_types[0])
    if not isinstance(message, Mapping):
        errors.append(f"{message_types[0]} must be an object")
        return None

    return message_types[0]


def _validate_with_a2ui_sdk(
    payload: Mapping[str, Any],
    errors: list[str],
) -> None:
    try:
        _sdk_basic_catalog_validator().validate(
            _sdk_compatible_payload(payload),
            strict_integrity=False,
        )
    except Exception as exc:
        errors.append(_format_sdk_validation_error(exc))


def _sdk_compatible_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        normalized: dict[Any, Any] = {}
        for key, child_value in value.items():
            if key == "context" and isinstance(child_value, Mapping):
                normalized[key] = _sdk_compatible_event_context(child_value)
            else:
                normalized[key] = _sdk_compatible_payload(child_value)
        return normalized

    if isinstance(value, list):
        return [_sdk_compatible_payload(item) for item in value]

    return value


def _sdk_compatible_event_context(context: Mapping[str, Any]) -> dict[Any, Any]:
    normalized = {
        key: _sdk_compatible_payload(value)
        for key, value in context.items()
    }
    if (
        isinstance(context.get("type"), str)
        and isinstance(context.get("surfaceId"), str)
        and isinstance(context.get("payload"), Mapping)
    ):
        # The v0.9 SDK schema currently limits event.context values to scalar,
        # array, or data-binding values. Approval buttons intentionally carry
        # the local A2UI userAction payload object so the renderer can dispatch
        # the event directly into the inbound parser.
        normalized["payload"] = []
    return normalized


@lru_cache(maxsize=1)
def _sdk_basic_catalog_validator() -> Any:
    schema_manager = A2uiSchemaManager(
        VERSION_0_9,
        [SdkBasicCatalog.get_config(VERSION_0_9)],
    )
    return schema_manager.get_selected_catalog().validator


def _format_sdk_validation_error(exc: Exception) -> str:
    message = str(exc).strip()
    if not message:
        message = type(exc).__name__
    return f"A2UI SDK validation failed: {message}"


def _validate_create_surface(value: Any, errors: list[str]) -> None:
    if not isinstance(value, Mapping):
        errors.append(f"{CREATE_SURFACE_MESSAGE} must be an object")
        return

    _require_pattern(
        value,
        "surfaceId",
        SURFACE_ID_PATTERN,
        errors,
        path=CREATE_SURFACE_MESSAGE,
    )
    _require_string(
        value,
        "catalogId",
        errors,
        path=CREATE_SURFACE_MESSAGE,
        expected_value=BASIC_CATALOG_ID,
    )


def _validate_surface_message(
    value: Any,
    message_type: str,
    errors: list[str],
) -> None:
    if not isinstance(value, Mapping):
        errors.append(f"{message_type} must be an object")
        return
    _require_pattern(value, "surfaceId", SURFACE_ID_PATTERN, errors, path=message_type)


def _validate_approval_canvas_payload(
    payload: Mapping[str, Any],
    components: list[Any],
    errors: list[str],
) -> None:
    _require_string(
        payload,
        "catalog",
        errors,
        expected_value=BASIC_CATALOG_NAME,
    )
    _require_pattern(payload, "planId", PLAN_ID_PATTERN, errors)
    _require_positive_int(payload, "planVersion", errors)
    _require_string(
        payload,
        "kind",
        errors,
        expected_value=WORKFLOW_CANVAS_TYPE,
    )

    workflow_canvas_seen = False
    for index, component in enumerate(components):
        path = f"components[{index}]"
        if not isinstance(component, Mapping):
            errors.append(f"{path} must be an object")
            continue
        component_type = component.get("type")
        if component_type == WORKFLOW_CANVAS_TYPE:
            workflow_canvas_seen = True
            _validate_workflow_canvas_component(component, path, payload, errors)
        elif not isinstance(component_type, str) or not component_type:
            errors.append(f"{path}.type must be a non-empty string")
        else:
            errors.append(
                f"{path}.type {component_type!r} is not supported for "
                "approval canvases"
            )

    if not workflow_canvas_seen:
        errors.append("components must include a workflowCanvas component")


def _validate_generic_components(
    components: list[Any],
    errors: list[str],
    *,
    path: str = "components",
) -> None:
    for index, component in enumerate(components):
        _validate_generic_component(component, f"{path}[{index}]", errors)


def _validate_generic_component(
    component: Any,
    path: str,
    errors: list[str],
) -> None:
    if not isinstance(component, Mapping):
        errors.append(f"{path} must be an object")
        return

    component_type = component.get("type", component.get("component"))
    if not isinstance(component_type, str) or not component_type:
        errors.append(
            f"{path}.type or {path}.component "
            "must be a non-empty string"
        )
        return

    if component_type not in SUPPORTED_BASIC_COMPONENT_TYPES:
        errors.append(
            f"{path}.type {component_type!r} must be a supported Basic "
            "Catalog component"
        )
        return

    component_type_key = component_type.casefold()
    _validate_generic_component_shape(
        component,
        path,
        component_type_key,
        errors,
    )


def validate_basic_catalog_payload(payload: Mapping[str, Any]) -> list[str]:
    """Return schema validation errors for a Basic Catalog payload."""

    return BasicCatalogSchema().validate(payload)


def _validate_generic_component_shape(
    component: Mapping[str, Any],
    path: str,
    component_type: str,
    errors: list[str],
) -> None:
    _require_string(component, "id", errors, path=path)

    if component_type == "audioplayer":
        _require_non_empty_field(component, "url", errors, path=path)
    elif component_type == "button":
        _require_label_or_child(component, path, errors)
        _validate_button_action(component.get("action"), f"{path}.action", errors)
    elif component_type == "card":
        _require_card_content(component, path, errors)
    elif component_type == "checkbox":
        _require_non_empty_field(component, "label", errors, path=path)
        _require_present_field(component, "value", errors, path=path)
    elif component_type == "choicepicker":
        _validate_choice_options(component.get("options"), f"{path}.options", errors)
        _require_present_field(component, "value", errors, path=path)
    elif component_type in {"column", "list", "row"}:
        _validate_child_components(component.get("children"), f"{path}.children", errors)
    elif component_type == "datetimeinput":
        _require_present_field(component, "value", errors, path=path)
    elif component_type == "icon":
        _require_non_empty_field(component, "name", errors, path=path)
    elif component_type == "image":
        _require_non_empty_field(component, "url", errors, path=path)
    elif component_type == "modal":
        _require_non_empty_field(component, "trigger", errors, path=path)
        _require_non_empty_field(component, "content", errors, path=path)
    elif component_type == "slider":
        _require_present_field(component, "value", errors, path=path)
        _require_present_field(component, "max", errors, path=path)
    elif component_type == "tabs":
        _validate_tabs(component.get("tabs"), f"{path}.tabs", errors)
    elif component_type == "text":
        _require_non_empty_field(component, "text", errors, path=path)
    elif component_type == "textfield":
        _require_non_empty_field(component, "label", errors, path=path)
    elif component_type == "video":
        _require_non_empty_field(component, "url", errors, path=path)


def _require_label_or_child(
    component: Mapping[str, Any],
    path: str,
    errors: list[str],
) -> None:
    if _has_non_empty_value(component.get("label")) or _has_non_empty_value(
        component.get("child")
    ):
        return

    errors.append(f"{path}.label or {path}.child must be present for button")


def _validate_button_action(value: Any, path: str, errors: list[str]) -> None:
    if not isinstance(value, Mapping):
        errors.append(f"{path} must be an object")
        return

    event = value.get("event")
    if not isinstance(event, Mapping):
        return

    plan_action_event = _is_plan_action_event(event, None)
    context = event.get("context")
    if not isinstance(context, Mapping):
        if plan_action_event:
            errors.append(f"{path}.event.context must be an object for plan action")
        return

    plan_action_event = _is_plan_action_event(event, context)
    if not _has_structured_user_action_context(context):
        if plan_action_event:
            _require_user_action_context_fields(
                context,
                f"{path}.event.context",
                errors,
            )
        return

    action_type = context["type"]
    if action_type not in ALLOWED_CONTROL_ACTIONS:
        errors.append(f"{path}.event.context.type must be a supported plan action")

    _require_pattern(
        context,
        "surfaceId",
        SURFACE_ID_PATTERN,
        errors,
        path=f"{path}.event.context",
    )

    payload = context.get("payload")
    if not isinstance(payload, Mapping):
        errors.append(f"{path}.event.context.payload must be an object")
        return

    if not str(context.get("surfaceId")).startswith(PLAN_APPROVAL_SURFACE_PREFIX):
        return

    _require_pattern(
        payload,
        "planId",
        PLAN_ID_PATTERN,
        errors,
        path=f"{path}.event.context.payload",
    )
    if action_type != "reject_plan":
        _require_positive_int(
            payload,
            "planVersion",
            errors,
            path=f"{path}.event.context.payload",
        )


def _is_plan_action_event(
    event: Mapping[str, Any],
    context: Mapping[str, Any] | None,
) -> bool:
    event_name = event.get("name")
    if isinstance(event_name, str) and event_name in ALLOWED_CONTROL_ACTIONS:
        return True

    if context is None:
        return False

    context_type = context.get("type")
    if isinstance(context_type, str) and context_type in ALLOWED_CONTROL_ACTIONS:
        return True

    surface_id = context.get("surfaceId")
    return (
        isinstance(surface_id, str)
        and surface_id.startswith(PLAN_APPROVAL_SURFACE_PREFIX)
    )


def _has_structured_user_action_context(context: Mapping[str, Any]) -> bool:
    return (
        isinstance(context.get("type"), str)
        and isinstance(context.get("surfaceId"), str)
        and "payload" in context
    )


def _require_user_action_context_fields(
    context: Mapping[str, Any],
    path: str,
    errors: list[str],
) -> None:
    if not isinstance(context.get("type"), str):
        errors.append(f"{path}.type must be a non-empty string")
    if not isinstance(context.get("surfaceId"), str):
        errors.append(f"{path}.surfaceId must be a non-empty string")
    if "payload" not in context:
        errors.append(f"{path}.payload must be present")


def _require_card_content(
    component: Mapping[str, Any],
    path: str,
    errors: list[str],
) -> None:
    if (
        _has_non_empty_value(component.get("child"))
        or _has_non_empty_value(component.get("title"))
        or _has_non_empty_value(component.get("body"))
    ):
        return

    errors.append(f"{path}.child, {path}.title, or {path}.body must be present")


def _validate_choice_options(value: Any, path: str, errors: list[str]) -> None:
    if not isinstance(value, list) or not value:
        errors.append(f"{path} must be a non-empty list")
        return

    for index, option in enumerate(value):
        option_path = f"{path}[{index}]"
        if not isinstance(option, Mapping):
            errors.append(f"{option_path} must be an object")
            continue
        _require_non_empty_field(option, "label", errors, path=option_path)
        _require_non_empty_field(option, "value", errors, path=option_path)


def _validate_child_components(value: Any, path: str, errors: list[str]) -> None:
    if isinstance(value, Mapping):
        _validate_child_template(value, path, errors)
        return

    if not isinstance(value, list) or not value:
        errors.append(f"{path} must be a non-empty list")
        return

    for index, child in enumerate(value):
        child_path = f"{path}[{index}]"
        if isinstance(child, str):
            if not child:
                errors.append(
                    f"{child_path} must be a non-empty component ID string"
                )
            continue

        if isinstance(child, Mapping):
            _validate_generic_component(child, child_path, errors)
            continue

        errors.append(f"{child_path} must be a component ID string or object")


def _validate_child_template(
    value: Mapping[str, Any],
    path: str,
    errors: list[str],
) -> None:
    allowed_keys = {"componentId", "path"}
    extra_keys = sorted(set(value) - allowed_keys)
    if extra_keys:
        errors.append(f"{path} child template has unsupported keys")

    component_id = value.get("componentId")
    if not isinstance(component_id, str) or not component_id:
        errors.append(f"{path}.componentId must be a non-empty component ID string")

    if not isinstance(value.get("path"), str):
        errors.append(f"{path}.path must be a string")


def _validate_component_graph(
    components: list[Any],
    errors: list[str],
    *,
    path: str,
) -> None:
    component_ids: set[str] = set()
    for index, component in enumerate(components):
        component_path = f"{path}[{index}]"
        if not isinstance(component, Mapping):
            continue

        component_id = component.get("id")
        if not isinstance(component_id, str) or not component_id:
            continue
        if component_id in component_ids:
            errors.append(f"{component_path}.id must be unique")
            continue
        component_ids.add(component_id)

    if "root" not in component_ids:
        errors.append(f"{path} must include a component with id 'root'")

    for index, component in enumerate(components):
        if not isinstance(component, Mapping):
            continue
        _validate_component_references(
            component,
            component_ids,
            f"{path}[{index}]",
            errors,
        )


def _validate_component_references(
    component: Mapping[str, Any],
    component_ids: set[str],
    path: str,
    errors: list[str],
) -> None:
    for field_name in ("child", "trigger", "content"):
        _validate_component_id_reference(
            component.get(field_name),
            component_ids,
            f"{path}.{field_name}",
            errors,
        )

    children = component.get("children")
    if isinstance(children, list):
        for index, child_id in enumerate(children):
            _validate_component_id_reference(
                child_id,
                component_ids,
                f"{path}.children[{index}]",
                errors,
            )
    elif isinstance(children, Mapping):
        _validate_component_id_reference(
            children.get("componentId"),
            component_ids,
            f"{path}.children.componentId",
            errors,
        )

    tabs = component.get("tabs")
    if isinstance(tabs, list):
        for index, tab in enumerate(tabs):
            if isinstance(tab, Mapping):
                _validate_component_id_reference(
                    tab.get("child"),
                    component_ids,
                    f"{path}.tabs[{index}].child",
                    errors,
                )


def _validate_component_id_reference(
    value: Any,
    component_ids: set[str],
    path: str,
    errors: list[str],
) -> None:
    if value is None:
        return
    if not isinstance(value, str) or not value:
        errors.append(f"{path} must be a component ID string")
        return
    if value not in component_ids:
        errors.append(f"{path} references unknown component {value!r}")


def _validate_tabs(value: Any, path: str, errors: list[str]) -> None:
    if not isinstance(value, list) or not value:
        errors.append(f"{path} must be a non-empty list")
        return

    for index, tab in enumerate(value):
        tab_path = f"{path}[{index}]"
        if not isinstance(tab, Mapping):
            errors.append(f"{tab_path} must be an object")
            continue
        _require_non_empty_field(tab, "title", errors, path=tab_path)
        _require_non_empty_field(tab, "child", errors, path=tab_path)


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
            if isinstance(dependency, str) and dependency not in declared_step_ids
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
        if not isinstance(action_type, str) or action_type not in ALLOWED_CONTROL_ACTIONS:
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


def _require_non_empty_field(
    payload: Mapping[str, Any],
    field_name: str,
    errors: list[str],
    *,
    path: str,
) -> None:
    value = payload.get(field_name)
    if not _has_non_empty_value(value):
        errors.append(f"{path}.{field_name} must be present")


def _require_present_field(
    payload: Mapping[str, Any],
    field_name: str,
    errors: list[str],
    *,
    path: str,
) -> None:
    if field_name not in payload or payload.get(field_name) is None:
        errors.append(f"{path}.{field_name} must be present")


def _require_mapping_field(
    payload: Mapping[str, Any],
    field_name: str,
    errors: list[str],
    *,
    path: str,
) -> None:
    value = payload.get(field_name)
    if not isinstance(value, Mapping):
        errors.append(f"{path}.{field_name} must be an object")


def _require_non_empty_list_field(
    payload: Mapping[str, Any],
    field_name: str,
    errors: list[str],
    *,
    path: str,
) -> None:
    value = payload.get(field_name)
    if not isinstance(value, list) or not value:
        errors.append(f"{path}.{field_name} must be a non-empty list")


def _has_non_empty_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str | list | tuple | set | frozenset | Mapping):
        return bool(value)
    return True


def _require_pattern(
    payload: Mapping[str, Any],
    field_name: str,
    pattern: re.Pattern[str],
    errors: list[str],
    *,
    path: str | None = None,
) -> None:
    value = payload.get(field_name)
    field_path = f"{path}.{field_name}" if path else field_name
    if not isinstance(value, str) or not pattern.match(value):
        errors.append(f"{field_path} must match {pattern.pattern}")


def _require_positive_int(
    payload: Mapping[str, Any],
    field_name: str,
    errors: list[str],
    *,
    path: str | None = None,
) -> None:
    value = payload.get(field_name)
    field_path = f"{path}.{field_name}" if path else field_name
    if not isinstance(value, int) or value < 1:
        errors.append(f"{field_path} must be a positive integer")


__all__ = [
    "ALLOWED_CONTROL_ACTIONS",
    "BASIC_CATALOG_NAME",
    "BasicCatalogSchema",
    "WORKFLOW_CANVAS_TYPE",
    "validate_basic_catalog_payload",
]
