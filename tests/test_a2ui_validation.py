import json
from typing import Any

import pytest

from orchestrator_demo.a2a_support.transport import A2UI_MIME_TYPE, DataPart, TextPart
from orchestrator_demo.a2ui_support.schema_manager import (
    A2UI_VERSION,
    BASIC_CATALOG_ID,
)


def _a2ui_update(
    *,
    surface_id: str = "surface_product_card",
    components: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "version": A2UI_VERSION,
        "updateComponents": {
            "surfaceId": surface_id,
            "components": components
            if components is not None
            else [
                {
                    "component": "Text",
                    "id": "root",
                    "text": "Treasury services fit the stated need.",
                }
            ],
        },
    }


def _valid_canvas_payload() -> dict[str, Any]:
    return _a2ui_update(
        surface_id="surface_plan_meeting_prep",
        components=[
            {
                "component": "Column",
                "id": "root",
                "children": [
                    "component_plan_meeting_prep_metadata",
                    "component_plan_meeting_prep_objective",
                    "component_plan_meeting_prep_agents",
                    "component_plan_meeting_prep_steps",
                    "component_plan_meeting_prep_dependencies",
                    "component_plan_meeting_prep_parallel_groups",
                    "component_plan_meeting_prep_controls",
                ],
            },
            {
                "component": "Text",
                "id": "component_plan_meeting_prep_metadata",
                "text": (
                    "surfaceId: surface_plan_meeting_prep\n"
                    "planId: plan_meeting_prep\n"
                    "planVersion: 1"
                ),
            },
            {
                "component": "Text",
                "id": "component_plan_meeting_prep_objective",
                "text": "Objective: Prepare for the ABC Manufacturing meeting.",
            },
            {
                "component": "Text",
                "id": "component_plan_meeting_prep_agents",
                "text": "Selected agents: internal_knowledge",
            },
            {
                "component": "Text",
                "id": "component_plan_meeting_prep_steps",
                "text": "Steps: step_internal_knowledge",
            },
            {
                "component": "Text",
                "id": "component_plan_meeting_prep_dependencies",
                "text": "Dependencies: none",
            },
            {
                "component": "Text",
                "id": "component_plan_meeting_prep_parallel_groups",
                "text": "Parallel groups: parallel_context",
            },
            {
                "component": "Row",
                "id": "component_plan_meeting_prep_controls",
                "children": ["control_approve_plan"],
            },
            {
                "component": "Button",
                "id": "control_approve_plan",
                "child": "control_approve_plan_label",
                "variant": "primary",
                "action": {
                    "event": {
                        "name": "approve_plan",
                        "context": {
                            "type": "approve_plan",
                            "surfaceId": "surface_plan_meeting_prep",
                            "payload": {
                                "planId": "plan_meeting_prep",
                                "planVersion": 1,
                                "editedPlanVersion": 1,
                                "approvedStepIds": ["step_internal_knowledge"],
                            },
                        },
                    }
                },
            },
            {
                "component": "Text",
                "id": "control_approve_plan_label",
                "text": "Approve",
            },
        ],
    )


def _valid_workflow_canvas_payload() -> dict[str, Any]:
    return {
        "version": A2UI_VERSION,
        "updateComponents": {
            "surfaceId": "surface_plan_meeting_prep",
            "catalog": "basic",
            "planId": "plan_meeting_prep",
            "planVersion": 1,
            "kind": "workflowCanvas",
            "components": [
                {
                    "type": "workflowCanvas",
                    "id": "root",
                    "objective": "Prepare for the ABC Manufacturing meeting.",
                    "selectedAgents": [
                        {
                            "agentId": "internal_knowledge",
                            "displayName": "Internal Knowledge Agent",
                        }
                    ],
                    "steps": [
                        {
                            "stepId": "step_internal_knowledge",
                            "agentId": "internal_knowledge",
                            "instruction": "Review internal CRM notes.",
                            "dependsOn": [],
                            "expectedOutput": "Internal customer context.",
                        }
                    ],
                    "parallelGroups": [],
                    "controls": [
                        {
                            "controlId": "control_approve_plan",
                            "type": "approve_plan",
                            "label": "Approve",
                            "action": {
                                "type": "approve_plan",
                                "surfaceId": "surface_plan_meeting_prep",
                                "planId": "plan_meeting_prep",
                                "planVersion": 1,
                                "payload": {
                                    "planId": "plan_meeting_prep",
                                    "planVersion": 1,
                                    "approvedStepIds": ["step_internal_knowledge"],
                                },
                            },
                        }
                    ],
                }
            ],
        },
    }


def test_a2ui_validation_success_emits_data_part_without_repair() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui

    payload = _valid_canvas_payload()

    # Act
    result = validate_outbound_a2ui(payload)

    # Assert
    assert result.valid is True
    assert result.repaired is False
    assert result.validation_errors == []
    assert isinstance(result.renderer_part, DataPart)
    assert result.renderer_part.metadata["mimeType"] == A2UI_MIME_TYPE
    assert result.renderer_part.data == payload


def test_a2ui_validation_accepts_structured_workflow_canvas_payload() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui

    payload = _valid_workflow_canvas_payload()

    # Act
    result = validate_outbound_a2ui(payload)

    # Assert
    assert result.valid is True
    assert result.validation_errors == []
    assert isinstance(result.renderer_part, DataPart)
    assert result.renderer_part.data == payload


def test_a2ui_validation_accepts_json_string_payload() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui

    payload = _a2ui_update()

    # Act
    result = validate_outbound_a2ui(json.dumps(payload))

    # Assert
    assert result.valid is True
    assert result.validation_errors == []
    assert isinstance(result.renderer_part, DataPart)
    assert result.renderer_part.metadata["mimeType"] == A2UI_MIME_TYPE
    assert result.renderer_part.data == payload


def test_a2ui_validation_accepts_json_bytes_payload() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui

    payload = _a2ui_update()

    # Act
    result = validate_outbound_a2ui(json.dumps(payload).encode("utf-8"))

    # Assert
    assert result.valid is True
    assert result.validation_errors == []
    assert isinstance(result.renderer_part, DataPart)
    assert result.renderer_part.metadata["mimeType"] == A2UI_MIME_TYPE
    assert result.renderer_part.data == payload


def test_a2ui_validation_accepts_sdk_kind_data_part_payload() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui

    payload = _valid_canvas_payload()
    sdk_data_part = {
        "kind": "data",
        "mimeType": A2UI_MIME_TYPE,
        "data": payload,
    }

    # Act
    result = validate_outbound_a2ui(sdk_data_part)

    # Assert
    assert result.valid is True
    assert result.validation_errors == []
    assert isinstance(result.renderer_part, DataPart)
    assert result.renderer_part.data == payload
    assert result.renderer_part.metadata["mimeType"] == A2UI_MIME_TYPE


def test_a2ui_validation_redacts_secret_like_sdk_data_part_error_locations() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui

    sdk_data_part = {
        "kind": "data",
        "mimeType": A2UI_MIME_TYPE,
        "data": _valid_canvas_payload(),
        "OPENROUTER_API_KEY": "accidentally copied envelope key",
        "Authorization": "accidentally copied header key",
    }

    # Act
    result = validate_outbound_a2ui(sdk_data_part)

    # Assert
    assert result.valid is False
    assert isinstance(result.renderer_part, TextPart)
    exposed_diagnostic_text = repr(
        {
            "validation_errors": result.validation_errors,
            "diagnostics": result.renderer_part.metadata["developerDiagnostic"],
        }
    )
    assert "OPENROUTER_API_KEY" not in exposed_diagnostic_text
    assert "Authorization" not in exposed_diagnostic_text
    assert "<redacted-key>" in exposed_diagnostic_text


def test_a2ui_validation_accepts_sdk_created_a2ui_part_instance() -> None:
    # Arrange
    from a2ui.a2a.parts import create_a2ui_part

    from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui

    payload = _a2ui_update()
    sdk_part = create_a2ui_part(payload)

    # Act
    result = validate_outbound_a2ui(sdk_part)

    # Assert
    assert result.valid is True
    assert result.validation_errors == []
    assert isinstance(result.renderer_part, DataPart)
    assert result.renderer_part.data == payload
    assert result.renderer_part.metadata["mimeType"] == A2UI_MIME_TYPE


def test_a2ui_validation_preserves_create_surface_envelope() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui

    payload = {
        "version": A2UI_VERSION,
        "createSurface": {
            "surfaceId": "surface_product_card",
            "catalogId": BASIC_CATALOG_ID,
        },
    }

    # Act
    result = validate_outbound_a2ui(payload)

    # Assert
    assert result.valid is True
    assert isinstance(result.renderer_part, DataPart)
    assert result.renderer_part.data == payload


def test_a2ui_validation_preserves_non_plan_specialist_a2ui_payload() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui

    payload = _a2ui_update()

    # Act
    result = validate_outbound_a2ui(payload)

    # Assert
    assert result.valid is True
    assert result.validation_errors == []
    assert isinstance(result.renderer_part, DataPart)
    assert result.renderer_part.data == payload


def test_a2ui_validation_accepts_renderer_mapped_downstream_components() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui

    payload = _a2ui_update(
        surface_id="surface_product_downstream",
        components=[
            {
                "component": "Table",
                "id": "component_table",
                "columns": [{"key": "product", "label": "Product"}],
                "rows": [{"product": "Treasury services"}],
            },
            {
                "component": "Accordion",
                "id": "component_accordion",
                "title": "Rationale",
                "children": ["component_status"],
            },
            {
                "component": "Timeline",
                "id": "component_timeline",
                "items": [
                    {
                        "label": "Review",
                        "detail": "RM validates synthetic demo findings.",
                    }
                ],
            },
            {
                "component": "Status",
                "id": "component_status",
                "message": "Ready for RM review.",
            },
        ],
    )

    # Act
    result = validate_outbound_a2ui(payload)

    # Assert
    assert result.valid is True
    assert result.validation_errors == []
    assert isinstance(result.renderer_part, DataPart)
    assert result.renderer_part.data == payload


def test_a2ui_validation_rejects_simplified_top_level_component_payload() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui

    payload = {
        "surfaceId": "surface_product_card",
        "components": [
            {
                "component": "Text",
                "id": "root",
                "text": "This is not an A2UI server-to-client envelope.",
            }
        ],
    }

    # Act
    result = validate_outbound_a2ui(payload)

    # Assert
    assert result.valid is False
    assert isinstance(result.renderer_part, TextPart)
    assert any(
        "server-to-client message" in error for error in result.validation_errors
    )


def test_a2ui_validation_accepts_basic_catalog_child_id_references() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui

    payload = _a2ui_update(
        surface_id="surface_product_layout",
        components=[
            {
                "component": "Column",
                "id": "root",
                "children": [
                    "component_product_header",
                    "component_product_detail",
                ],
            },
            {
                "component": "Text",
                "id": "component_product_header",
                "text": "Product opportunities",
            },
            {
                "component": "Text",
                "id": "component_product_detail",
                "text": "Treasury and merchant services fit the stated need.",
            },
        ],
    )

    # Act
    result = validate_outbound_a2ui(payload)

    # Assert
    assert result.valid is True
    assert result.validation_errors == []
    assert isinstance(result.renderer_part, DataPart)
    assert result.renderer_part.data == payload


def test_a2ui_validation_accepts_basic_catalog_templated_child_list() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui

    payload = _a2ui_update(
        surface_id="surface_product_list",
        components=[
            {
                "component": "List",
                "id": "root",
                "children": {
                    "componentId": "component_product_item",
                    "path": "items",
                },
            },
            {
                "component": "Text",
                "id": "component_product_item",
                "text": {"path": "name"},
            },
        ],
    )

    # Act
    result = validate_outbound_a2ui(payload)

    # Assert
    assert result.valid is True
    assert result.validation_errors == []
    assert isinstance(result.renderer_part, DataPart)
    assert result.renderer_part.data == payload


def test_a2ui_validation_accepts_renderer_supported_basic_catalog_components() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui

    payload = _a2ui_update(
        surface_id="surface_product_insights",
        components=[
            {
                "component": "Column",
                "id": "root",
                "children": [
                    "component_status",
                    "component_table",
                    "component_accordion",
                    "component_timeline",
                ],
            },
            {
                "component": "Status",
                "id": "component_status",
                "status": "ready",
                "message": "Specialist output ready.",
            },
            {
                "component": "Table",
                "id": "component_table",
                "columns": [
                    {"key": "balance", "label": "Balance"},
                    {"key": "active", "label": "Active"},
                ],
                "rows": [
                    {"balance": 0, "active": False},
                ],
            },
            {
                "component": "Accordion",
                "id": "component_accordion",
                "title": "Assumptions",
                "children": ["component_assumptions"],
            },
            {
                "component": "Text",
                "id": "component_assumptions",
                "text": "Synthetic demo data only.",
            },
            {
                "component": "Timeline",
                "id": "component_timeline",
                "items": [
                    {
                        "label": "Qualified",
                        "detail": "Relationship summary reviewed.",
                    }
                ],
            },
        ],
    )

    # Act
    result = validate_outbound_a2ui(payload)

    # Assert
    assert result.valid is True
    assert result.validation_errors == []
    assert isinstance(result.renderer_part, DataPart)
    assert result.renderer_part.data == payload


@pytest.mark.parametrize(
    ("component", "expected_error"),
    [
        (
            {
                "component": "Button",
                "id": "root",
                "label": {"text": "Inspect"},
                "action": {
                    "event": {
                        "name": "inspect_product",
                        "context": {
                            "type": "inspect_product",
                            "surfaceId": "surface_mixed_invalid_basic",
                            "payload": {"productId": "treasury"},
                        },
                    },
                },
            },
            "updateComponents.components[1].label must be a non-empty string",
        ),
        (
            {
                "component": "TextField",
                "id": "root",
                "label": {"text": "Notes"},
            },
            "updateComponents.components[1].label must be a non-empty string",
        ),
    ],
)
def test_a2ui_validation_rejects_malformed_basic_component_in_mixed_payload(
    component: dict[str, Any],
    expected_error: str,
) -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui

    payload = _a2ui_update(
        surface_id="surface_mixed_invalid_basic",
        components=[
            {
                "component": "Table",
                "id": "component_table",
                "columns": [{"key": "name", "label": "Name"}],
                "rows": [{"name": "ABC Manufacturing"}],
            },
            component,
        ],
    )

    # Act
    result = validate_outbound_a2ui(payload)

    # Assert
    assert result.valid is False
    assert isinstance(result.renderer_part, TextPart)
    assert expected_error in result.validation_errors


def test_a2ui_validation_accepts_inline_child_extension_component() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui

    payload = _a2ui_update(
        surface_id="surface_inline_extension",
        components=[
            {
                "component": "Card",
                "id": "root",
                "child": {
                    "component": "Table",
                    "id": "component_inline_table",
                    "columns": [{"key": "name", "label": "Name"}],
                    "rows": [{"name": "ABC Manufacturing"}],
                },
            }
        ],
    )

    # Act
    result = validate_outbound_a2ui(payload)

    # Assert
    assert result.valid is True
    assert result.validation_errors == []
    assert isinstance(result.renderer_part, DataPart)
    assert result.renderer_part.data == payload


def test_a2ui_validation_rejects_duplicate_ids_in_mixed_extension_payload() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui

    payload = _a2ui_update(
        surface_id="surface_mixed_duplicate_ids",
        components=[
            {
                "component": "Table",
                "id": "component_table",
                "columns": [{"key": "name", "label": "Name"}],
                "rows": [{"name": "ABC Manufacturing"}],
            },
            {
                "component": "Text",
                "id": "component_duplicate",
                "text": "First summary.",
            },
            {
                "component": "Text",
                "id": "component_duplicate",
                "text": "Second summary.",
            },
        ],
    )

    # Act
    result = validate_outbound_a2ui(payload)

    # Assert
    assert result.valid is False
    assert isinstance(result.renderer_part, TextPart)
    assert any("id must be unique" in error for error in result.validation_errors)


def test_a2ui_validation_rejects_invalid_bindings_in_mixed_extension_payload() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui

    payload = _a2ui_update(
        surface_id="surface_mixed_invalid_binding",
        components=[
            {
                "component": "Table",
                "id": "component_table",
                "columns": [{"key": "name", "label": "Name"}],
                "rows": [{"name": "ABC Manufacturing"}],
            },
            {
                "component": "Text",
                "id": "root",
                "text": {"path": "customer/~2bad"},
            },
        ],
    )

    # Act
    result = validate_outbound_a2ui(payload)

    # Assert
    assert result.valid is False
    assert isinstance(result.renderer_part, TextPart)
    assert any(
        "valid A2UI path syntax" in error for error in result.validation_errors
    )


def test_a2ui_validation_rejects_invalid_action_bindings_in_mixed_extension_payload() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui

    payload = _a2ui_update(
        surface_id="surface_mixed_invalid_action_binding",
        components=[
            {
                "component": "Table",
                "id": "component_table",
                "columns": [{"key": "name", "label": "Name"}],
                "rows": [{"name": "ABC Manufacturing"}],
            },
            {
                "component": "Button",
                "id": "root",
                "label": "Inspect customer",
                "action": {
                    "event": {
                        "name": "inspect_customer",
                        "context": {
                            "type": "specialist_action",
                            "surfaceId": "surface_mixed_invalid_action_binding",
                            "payload": {
                                "action": "inspect_customer",
                                "customerName": {"path": "customer/~2bad"},
                            },
                        },
                    },
                },
            },
        ],
    )

    # Act
    result = validate_outbound_a2ui(payload)

    # Assert
    assert result.valid is False
    assert isinstance(result.renderer_part, TextPart)
    assert any(
        "event.context.payload.customerName.path must use valid A2UI path syntax"
        in error
        for error in result.validation_errors
    )


def test_a2ui_validation_redacts_secret_like_action_payload_keys_in_diagnostics() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui

    secret_like_key = "openrouter_api_key"
    payload = _a2ui_update(
        surface_id="surface_mixed_secret_key_invalid_binding",
        components=[
            {
                "component": "Table",
                "id": "component_table",
                "columns": [{"key": "name", "label": "Name"}],
                "rows": [{"name": "ABC Manufacturing"}],
            },
            {
                "component": "Button",
                "id": "root",
                "label": "Inspect customer",
                "action": {
                    "event": {
                        "name": "inspect_customer",
                        "context": {
                            "type": "specialist_action",
                            "surfaceId": "surface_mixed_secret_key_invalid_binding",
                            "payload": {
                                secret_like_key: {"path": "customer/~2bad"},
                            },
                        },
                    },
                },
            },
        ],
    )

    # Act
    result = validate_outbound_a2ui(payload)

    # Assert
    assert result.valid is False
    assert isinstance(result.renderer_part, TextPart)
    exposed_diagnostic_text = repr(
        {
            "validation_errors": result.validation_errors,
            "diagnostics": result.renderer_part.metadata["developerDiagnostic"],
        }
    )
    assert secret_like_key not in exposed_diagnostic_text
    assert "payload.<redacted-key>.path must use valid A2UI path syntax" in (
        exposed_diagnostic_text
    )
    assert "A2UI payload contains secret-like key" in exposed_diagnostic_text


@pytest.mark.parametrize(
    ("missing_field", "expected_error"),
    [
        ("type", "action.event.context.type must be a non-empty string"),
        ("surfaceId", "action.event.context.surfaceId must be a non-empty string"),
    ],
)
def test_a2ui_validation_rejects_mixed_routed_button_without_user_action_context(
    missing_field: str,
    expected_error: str,
) -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui

    button = {
        "component": "Button",
        "id": "root",
        "label": "Inspect customer",
        "action": {
            "event": {
                "name": "inspect_customer",
                "context": {
                    "type": "specialist_action",
                    "surfaceId": "surface_mixed_invalid_button_action",
                    "payload": {"action": "inspect_customer"},
                },
            },
        },
    }
    button["action"]["event"]["context"].pop(missing_field)
    payload = _a2ui_update(
        surface_id="surface_mixed_invalid_button_action",
        components=[
            {
                "component": "Table",
                "id": "component_table",
                "columns": [{"key": "name", "label": "Name"}],
                "rows": [{"name": "ABC Manufacturing"}],
            },
            button,
        ],
    )

    # Act
    result = validate_outbound_a2ui(payload)

    # Assert
    assert result.valid is False
    assert isinstance(result.renderer_part, TextPart)
    assert any(expected_error in error for error in result.validation_errors)


def test_a2ui_validation_rejects_non_object_button_event_in_mixed_payload() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui

    payload = _a2ui_update(
        surface_id="surface_mixed_invalid_button_event",
        components=[
            {
                "component": "Table",
                "id": "component_table",
                "columns": [{"key": "name", "label": "Name"}],
                "rows": [{"name": "ABC Manufacturing"}],
            },
            {
                "component": "Button",
                "id": "root",
                "label": "Inspect customer",
                "action": {"event": "inspect_customer"},
            },
        ],
    )

    # Act
    result = validate_outbound_a2ui(payload)

    # Assert
    assert result.valid is False
    assert isinstance(result.renderer_part, TextPart)
    assert any(
        "updateComponents.components[1].action.event must be an object" in error
        for error in result.validation_errors
    )


@pytest.mark.parametrize(
    ("component", "expected_error"),
    [
        (
            {
                "component": "Text",
                "id": "root",
                "text": {"path": 123},
            },
            "updateComponents.components[1].text.path must be a string",
        ),
        (
            {
                "component": "TextField",
                "id": "root",
                "label": "Customer",
                "value": {"path": 123},
            },
            "updateComponents.components[1].value.path must be a string",
        ),
        (
            {
                "component": "Button",
                "id": "root",
                "label": "Inspect customer",
                "action": {
                    "event": {
                        "name": "inspect_customer",
                        "context": {
                            "type": "specialist_action",
                            "surfaceId": "surface_mixed_non_string_binding",
                            "payload": {
                                "action": "inspect_customer",
                                "customerName": {"path": 123},
                            },
                        },
                    },
                },
            },
            (
                "updateComponents.components[1].action.event.context.payload."
                "customerName.path must be a string"
            ),
        ),
    ],
)
def test_a2ui_validation_rejects_non_string_binding_paths_in_mixed_payload(
    component: dict[str, Any],
    expected_error: str,
) -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui

    payload = _a2ui_update(
        surface_id="surface_mixed_non_string_binding",
        components=[
            {
                "component": "Table",
                "id": "component_table",
                "columns": [{"key": "name", "label": "Name"}],
                "rows": [{"name": "ABC Manufacturing"}],
            },
            component,
        ],
    )

    # Act
    result = validate_outbound_a2ui(payload)

    # Assert
    assert result.valid is False
    assert isinstance(result.renderer_part, TextPart)
    assert expected_error in result.validation_errors


def test_a2ui_validation_allows_literal_path_fields_in_mixed_extension_payload() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui

    payload = _a2ui_update(
        surface_id="surface_mixed_literal_paths",
        components=[
            {
                "component": "Column",
                "id": "root",
                "children": ["component_documents", "component_open_report"],
            },
            {
                "component": "Table",
                "id": "component_documents",
                "columns": [
                    {"key": "name", "label": "Name"},
                    {"key": "path", "label": "Path"},
                ],
                "rows": [
                    {
                        "name": "Relationship summary",
                        "path": "~/docs/report.pdf",
                    }
                ],
            },
            {
                "component": "Button",
                "id": "component_open_report",
                "label": "Open report",
                "action": {
                    "event": {
                        "name": "open_report",
                        "context": {
                            "type": "open_report",
                            "surfaceId": "surface_mixed_literal_paths",
                            "payload": {
                                "path": "~/docs/report.pdf",
                                "label": "Relationship summary",
                            },
                        },
                    },
                },
            },
        ],
    )

    # Act
    result = validate_outbound_a2ui(payload)

    # Assert
    assert result.valid is True
    assert result.validation_errors == []
    assert isinstance(result.renderer_part, DataPart)
    assert result.renderer_part.data == payload


def test_a2ui_validation_allows_literal_single_key_path_rows_in_mixed_payload() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui

    payload = _a2ui_update(
        surface_id="surface_mixed_single_key_path_row",
        components=[
            {
                "component": "Table",
                "id": "root",
                "columns": [{"key": "path", "label": "Path"}],
                "rows": [{"path": "~/docs/report.pdf"}],
            },
        ],
    )

    # Act
    result = validate_outbound_a2ui(payload)

    # Assert
    assert result.valid is True
    assert result.validation_errors == []
    assert isinstance(result.renderer_part, DataPart)
    assert result.renderer_part.data == payload


def test_a2ui_validation_rejects_reference_cycles_in_mixed_extension_payload() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui

    payload = _a2ui_update(
        surface_id="surface_mixed_reference_cycle",
        components=[
            {
                "component": "Table",
                "id": "component_table",
                "columns": [{"key": "name", "label": "Name"}],
                "rows": [{"name": "ABC Manufacturing"}],
            },
            {
                "component": "Row",
                "id": "component_a",
                "children": ["component_b"],
            },
            {
                "component": "Row",
                "id": "component_b",
                "children": ["component_a"],
            },
        ],
    )

    # Act
    result = validate_outbound_a2ui(payload)

    # Assert
    assert result.valid is False
    assert isinstance(result.renderer_part, TextPart)
    assert any(
        "circular component references" in error for error in result.validation_errors
    )


def test_a2ui_validation_rejects_reference_cycles_in_basic_payload() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui

    payload = _a2ui_update(
        surface_id="surface_basic_reference_cycle",
        components=[
            {
                "component": "Column",
                "id": "root",
                "children": ["root"],
            },
        ],
    )

    # Act
    result = validate_outbound_a2ui(payload)

    # Assert
    assert result.valid is False
    assert isinstance(result.renderer_part, TextPart)
    assert any(
        "circular component references" in error
        or "must not reference itself" in error
        for error in result.validation_errors
    )


@pytest.mark.parametrize(
    ("component", "expected_error"),
    [
        (
            {
                "component": "Column",
                "id": "root",
                "children": ["component_table", "component_missing"],
            },
            (
                "updateComponents.components[1].children[1] references unknown "
                "component 'component_missing'"
            ),
        ),
        (
            {
                "component": "Card",
                "id": "root",
                "child": "component_missing",
            },
            (
                "updateComponents.components[1].child references unknown component "
                "'component_missing'"
            ),
        ),
        (
            {
                "component": "Modal",
                "id": "root",
                "trigger": "component_missing",
                "content": "component_table",
            },
            (
                "updateComponents.components[1].trigger references unknown "
                "component 'component_missing'"
            ),
        ),
        (
            {
                "component": "Tabs",
                "id": "root",
                "tabs": [
                    {
                        "title": "Overview",
                        "child": "component_missing",
                    }
                ],
            },
            (
                "updateComponents.components[1].tabs[0].child references unknown "
                "component 'component_missing'"
            ),
        ),
    ],
)
def test_a2ui_validation_rejects_unknown_references_in_mixed_payload(
    component: dict[str, Any],
    expected_error: str,
) -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui

    payload = _a2ui_update(
        surface_id="surface_mixed_unknown_reference",
        components=[
            {
                "component": "Table",
                "id": "component_table",
                "columns": [{"key": "name", "label": "Name"}],
                "rows": [{"name": "ABC Manufacturing"}],
            },
            component,
        ],
    )

    # Act
    result = validate_outbound_a2ui(payload)

    # Assert
    assert result.valid is False
    assert isinstance(result.renderer_part, TextPart)
    assert expected_error in result.validation_errors


def test_a2ui_validation_rejects_string_reference_to_inline_only_id() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui

    payload = _a2ui_update(
        surface_id="surface_mixed_inline_only_reference",
        components=[
            {
                "component": "Table",
                "id": "component_table",
                "columns": [{"key": "name", "label": "Name"}],
                "rows": [{"name": "ABC Manufacturing"}],
            },
            {
                "component": "Column",
                "id": "root",
                "children": [
                    {
                        "component": "Text",
                        "id": "component_inline_text",
                        "text": "Inline summary.",
                    },
                    "component_inline_text",
                ],
            },
        ],
    )

    # Act
    result = validate_outbound_a2ui(payload)

    # Assert
    assert result.valid is False
    assert isinstance(result.renderer_part, TextPart)
    assert (
        "updateComponents.components[1].children[1] references unknown "
        "component 'component_inline_text'"
    ) in result.validation_errors


def test_a2ui_validation_allows_incremental_reference_to_existing_component() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui

    surface_id = "surface_incremental_reference"
    payload = _a2ui_update(
        surface_id=surface_id,
        components=[
            {
                "component": "Table",
                "id": "component_table",
                "columns": [{"key": "name", "label": "Name"}],
                "rows": [{"name": "ABC Manufacturing"}],
            },
            {
                "component": "Column",
                "id": "root",
                "children": ["component_existing_summary"],
            },
        ],
    )
    existing_components = {
        surface_id: {
            "component_existing_summary": {
                "component": "Text",
                "id": "component_existing_summary",
                "text": "Existing summary.",
            }
        }
    }

    # Act
    result = validate_outbound_a2ui(
        payload,
        existing_components_by_surface_id=existing_components,
    )

    # Assert
    assert result.valid is True
    assert result.validation_errors == []


@pytest.mark.parametrize(
    "replacement_marker",
    [
        {"replace": True},
        {"fullReplacement": True},
        {"mode": "replace"},
    ],
)
def test_a2ui_validation_rejects_full_replacement_reference_to_existing_component(
    replacement_marker: dict[str, Any],
) -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui

    surface_id = "surface_full_replacement_reference"
    payload = _a2ui_update(
        surface_id=surface_id,
        components=[
            {
                "component": "Table",
                "id": "component_table",
                "columns": [{"key": "name", "label": "Name"}],
                "rows": [{"name": "ABC Manufacturing"}],
            },
            {
                "component": "Column",
                "id": "root",
                "children": ["component_existing_summary"],
            },
        ],
    )
    payload["updateComponents"].update(replacement_marker)
    existing_components = {
        surface_id: {
            "component_existing_summary": {
                "component": "Text",
                "id": "component_existing_summary",
                "text": "Existing summary.",
            }
        }
    }

    # Act
    result = validate_outbound_a2ui(
        payload,
        existing_components_by_surface_id=existing_components,
    )

    # Assert
    assert result.valid is False
    assert isinstance(result.renderer_part, TextPart)
    assert (
        "updateComponents.components[1].children[0] references unknown component "
        "'component_existing_summary'"
    ) in result.validation_errors


def test_a2ui_validation_rejects_inline_child_cycle_in_mixed_extension_payload() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui

    payload = _a2ui_update(
        surface_id="surface_mixed_inline_reference_cycle",
        components=[
            {
                "component": "Table",
                "id": "component_table",
                "columns": [{"key": "name", "label": "Name"}],
                "rows": [{"name": "ABC Manufacturing"}],
            },
            {
                "component": "Column",
                "id": "root",
                "children": [
                    {
                        "component": "Column",
                        "id": "inline_column",
                        "children": ["root"],
                    }
                ],
            },
        ],
    )

    # Act
    result = validate_outbound_a2ui(payload)

    # Assert
    assert result.valid is False
    assert isinstance(result.renderer_part, TextPart)
    assert any(
        "circular component references" in error for error in result.validation_errors
    )


def test_a2ui_validation_preserves_specialist_payload_with_plan_metadata() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui

    part = DataPart(
        data=_a2ui_update(),
        metadata={
            "mimeType": A2UI_MIME_TYPE,
            "planId": "plan_product_opportunity_abc123",
            "planVersion": 3,
        },
    )

    # Act
    result = validate_outbound_a2ui(part)

    # Assert
    assert result.valid is True
    assert result.validation_errors == []
    assert isinstance(result.renderer_part, DataPart)
    assert result.renderer_part.data == part.data
    assert result.renderer_part.metadata["planId"] == "plan_product_opportunity_abc123"


def test_a2ui_validation_falls_back_for_malformed_generic_component() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui

    payload = _a2ui_update(
        surface_id="surface_specialist_actions",
        components=[
            {
                "component": "Button",
                "id": "root",
            }
        ],
    )

    # Act
    result = validate_outbound_a2ui(payload)

    # Assert
    assert result.valid is False
    assert isinstance(result.renderer_part, TextPart)
    assert any("child" in error for error in result.validation_errors)
    assert any("action" in error for error in result.validation_errors)


@pytest.mark.parametrize(
    ("missing_field", "expected_error"),
    [
        ("context", "event.context must be an object for plan action"),
        ("type", "event.context.type must be a non-empty string"),
        ("surfaceId", "event.context.surfaceId must be a non-empty string"),
        ("payload", "event.context.payload must be present"),
    ],
)
def test_a2ui_validation_rejects_plan_action_button_without_user_action_context(
    missing_field: str,
    expected_error: str,
) -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui

    payload = _valid_canvas_payload()
    button = next(
        component
        for component in payload["updateComponents"]["components"]
        if component.get("id") == "control_approve_plan"
    )
    event = button["action"]["event"]
    if missing_field == "context":
        event.pop("context")
    else:
        event["context"].pop(missing_field)

    # Act
    result = validate_outbound_a2ui(payload)

    # Assert
    assert result.valid is False
    assert isinstance(result.renderer_part, TextPart)
    assert any(expected_error in error for error in result.validation_errors)


def test_a2ui_validation_recursively_rejects_unsupported_nested_component() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui

    payload = _a2ui_update(
        surface_id="surface_nested_components",
        components=[
            {
                "component": "Row",
                "id": "root",
                "children": [
                    {
                        "component": "Script",
                        "id": "component_script",
                    }
                ],
            }
        ],
    )

    # Act
    result = validate_outbound_a2ui(payload)

    # Assert
    assert result.valid is False
    assert isinstance(result.renderer_part, TextPart)
    assert any(
        "updateComponents.components[0].children[0].type 'Script' must be a "
        "supported Basic Catalog component" in error
        for error in result.validation_errors
    )


def test_a2ui_validation_preserves_partial_update_without_root() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui

    payload = _a2ui_update(
        components=[
            {
                "component": "Text",
                "id": "component_product_detail",
                "text": "Treasury and merchant services fit the stated need.",
            }
        ]
    )

    # Act
    result = validate_outbound_a2ui(payload)

    # Assert
    assert result.valid is True
    assert result.validation_errors == []
    assert isinstance(result.renderer_part, DataPart)
    assert result.renderer_part.data == payload


def test_a2ui_validation_preserves_partial_update_with_existing_child_reference() -> (
    None
):
    # Arrange
    from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui

    payload = _a2ui_update(
        components=[
            {
                "component": "Row",
                "id": "component_product_actions",
                "children": ["component_existing_summary"],
            }
        ]
    )

    # Act
    result = validate_outbound_a2ui(payload)

    # Assert
    assert result.valid is True
    assert result.validation_errors == []
    assert isinstance(result.renderer_part, DataPart)
    assert result.renderer_part.data == payload


def test_a2ui_validation_rejects_approval_canvas_update_without_root() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui

    payload = _a2ui_update(
        surface_id="surface_plan_product_card",
        components=[
            {
                "component": "Text",
                "id": "component_product_detail",
                "text": "Treasury and merchant services fit the stated need.",
            }
        ],
    )

    # Act
    result = validate_outbound_a2ui(payload)

    # Assert
    assert result.valid is False
    assert isinstance(result.renderer_part, TextPart)
    assert any(
        "must include a component with id 'root'" in error
        for error in result.validation_errors
    )


def test_a2ui_validation_retries_repair_once_and_emits_repaired_data_part() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui

    invalid_payload = _valid_canvas_payload()
    invalid_payload["updateComponents"].pop("surfaceId")
    repair_calls: list[tuple[dict[str, Any], list[str]]] = []

    def repair_once(payload: dict[str, Any], errors: list[str]) -> dict[str, Any]:
        repair_calls.append((payload, errors))
        repaired_payload = dict(payload)
        repaired_payload["updateComponents"] = {
            **payload["updateComponents"],
            "surfaceId": "surface_plan_meeting_prep",
        }
        return repaired_payload

    # Act
    result = validate_outbound_a2ui(invalid_payload, repair=repair_once)

    # Assert
    assert len(repair_calls) == 1
    assert "surfaceId" in repair_calls[0][1][0]
    assert result.valid is True
    assert result.repaired is True
    assert isinstance(result.renderer_part, DataPart)
    assert (
        result.renderer_part.data["updateComponents"]["surfaceId"]
        == "surface_plan_meeting_prep"
    )


def test_a2ui_validation_falls_back_to_text_after_failed_repair() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui

    invalid_payload = _valid_canvas_payload()
    invalid_payload["updateComponents"]["components"] = []
    repair_calls: list[list[str]] = []

    def failed_repair(payload: dict[str, Any], errors: list[str]) -> dict[str, Any]:
        repair_calls.append(errors)
        return payload

    # Act
    result = validate_outbound_a2ui(invalid_payload, repair=failed_repair)

    # Assert
    assert len(repair_calls) == 1
    assert result.valid is False
    assert result.repaired is False
    assert isinstance(result.renderer_part, TextPart)
    assert "A2UI rendering unavailable" in result.renderer_part.text
    assert result.renderer_part.metadata["developerDiagnostic"]["fallback"] == "text"
    assert result.renderer_part.metadata["developerDiagnostic"]["validationErrors"]
    assert not isinstance(result.renderer_part, DataPart)


def test_a2ui_validation_skips_repair_when_invalid_payload_contains_secret() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui

    leaked_value = "OPENROUTER_API_KEY=sk-live-invalid-a2ui-secret-token-123456789"
    invalid_payload = {
        "components": [
            {
                "component": "Text",
                "id": "root",
                "text": leaked_value,
            }
        ],
    }
    repair_calls: list[dict[str, Any]] = []

    def repair_should_not_run(
        payload: dict[str, Any],
        errors: list[str],
    ) -> dict[str, Any]:
        del errors
        repair_calls.append(payload)
        return payload

    # Act
    result = validate_outbound_a2ui(
        invalid_payload,
        repair=repair_should_not_run,
    )

    # Assert
    assert repair_calls == []
    assert result.valid is False
    assert isinstance(result.renderer_part, TextPart)
    diagnostics = result.renderer_part.metadata["developerDiagnostic"]
    exposed_diagnostic_text = repr(
        {
            "validation_errors": result.validation_errors,
            "diagnostics": diagnostics,
            "renderer_part": result.renderer_part.model_dump(mode="json"),
        }
    )
    assert diagnostics["repairAttempted"] is False
    assert leaked_value not in exposed_diagnostic_text
    assert "sk-live-invalid-a2ui-secret-token-123456789" not in exposed_diagnostic_text
    assert "<redacted-secret>" in exposed_diagnostic_text
    assert "secret-like value" in exposed_diagnostic_text


def test_a2ui_validation_redacts_secret_like_values_from_fallback_diagnostics() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui

    secret_component_id = "sk-live-a2ui-secret-token-123456789"
    invalid_payload = _valid_canvas_payload()
    invalid_payload["updateComponents"]["components"][0]["children"] = [
        secret_component_id
    ]

    # Act
    result = validate_outbound_a2ui(invalid_payload)

    # Assert
    assert result.valid is False
    assert isinstance(result.renderer_part, TextPart)
    diagnostics = result.renderer_part.metadata["developerDiagnostic"]
    exposed_diagnostic_text = repr(
        {
            "validation_errors": result.validation_errors,
            "diagnostics": diagnostics,
        }
    )
    assert secret_component_id not in exposed_diagnostic_text
    assert "<redacted-secret>" in exposed_diagnostic_text


def test_a2ui_validation_rejects_schema_valid_secret_bearing_payload() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui

    leaked_value = "OPENROUTER_API_KEY=sk-live-a2ui-secret-token-123456789"
    payload = _a2ui_update(
        components=[
            {
                "component": "Text",
                "id": "root",
                "text": leaked_value,
            }
        ]
    )

    # Act
    result = validate_outbound_a2ui(payload)

    # Assert
    assert result.valid is False
    assert isinstance(result.renderer_part, TextPart)
    diagnostics = result.renderer_part.metadata["developerDiagnostic"]
    exposed_diagnostic_text = repr(
        {
            "validation_errors": result.validation_errors,
            "diagnostics": diagnostics,
            "renderer_part": result.renderer_part.model_dump(mode="json"),
        }
    )
    assert not isinstance(result.renderer_part, DataPart)
    assert leaked_value not in exposed_diagnostic_text
    assert "sk-live-a2ui-secret-token-123456789" not in exposed_diagnostic_text
    assert "<redacted-secret>" in exposed_diagnostic_text
    assert "secret-like value" in exposed_diagnostic_text


@pytest.mark.parametrize(
    ("surface_id", "component_id", "leaked_fragment"),
    [
        (
            "surface_product_card",
            "component_sk-live-renderer-id-secret-token-123456789",
            "sk-live-renderer-id-secret-token-123456789",
        ),
        (
            "surface_ghp_abcdefghijklmnopqrstuvwxyz123456",
            "root",
            "ghp_abcdefghijklmnopqrstuvwxyz123456",
        ),
    ],
)
def test_a2ui_validation_rejects_secret_tokens_embedded_in_renderer_ids(
    surface_id: str,
    component_id: str,
    leaked_fragment: str,
) -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui

    payload = _a2ui_update(
        surface_id=surface_id,
        components=[
            {
                "component": "Text",
                "id": component_id,
                "text": "Treasury services fit the stated need.",
            }
        ],
    )

    # Act
    result = validate_outbound_a2ui(payload)

    # Assert
    assert result.valid is False
    assert isinstance(result.renderer_part, TextPart)
    assert not isinstance(result.renderer_part, DataPart)
    exposed_diagnostic_text = repr(
        {
            "validation_errors": result.validation_errors,
            "diagnostics": result.renderer_part.metadata["developerDiagnostic"],
            "renderer_part": result.renderer_part.model_dump(mode="json"),
        }
    )
    assert leaked_fragment not in exposed_diagnostic_text
    assert "<redacted-secret>" in exposed_diagnostic_text
    assert "secret-like value" in exposed_diagnostic_text


@pytest.mark.parametrize(
    "leaked_value",
    [
        "Authorization: Bearer renderer-credential-1234567890",
        "Bearer renderer-credential-1234567890",
    ],
)
def test_a2ui_validation_rejects_bearer_credentials_before_renderer(
    leaked_value: str,
) -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui

    payload = _a2ui_update(
        components=[
            {
                "component": "Text",
                "id": "root",
                "text": f"Debug header copied from upstream: {leaked_value}",
            }
        ]
    )

    # Act
    result = validate_outbound_a2ui(payload)

    # Assert
    assert result.valid is False
    assert isinstance(result.renderer_part, TextPart)
    assert not isinstance(result.renderer_part, DataPart)
    exposed_diagnostic_text = repr(
        {
            "validation_errors": result.validation_errors,
            "diagnostics": result.renderer_part.metadata["developerDiagnostic"],
            "renderer_part": result.renderer_part.model_dump(mode="json"),
        }
    )
    assert leaked_value not in exposed_diagnostic_text
    assert "renderer-credential-1234567890" not in exposed_diagnostic_text
    assert "<redacted-secret>" in exposed_diagnostic_text
    assert "secret-like value" in exposed_diagnostic_text


def test_a2ui_validation_rejects_repaired_secret_bearing_payload() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui

    leaked_value = "OPENROUTER_API_KEY=sk-live-repaired-secret-token-123456789"
    invalid_payload = _a2ui_update(
        components=[
            {
                "component": "Text",
                "id": "root",
            }
        ]
    )

    def repair_with_secret(
        payload: dict[str, Any], errors: list[str]
    ) -> dict[str, Any]:
        del errors
        repaired_payload = dict(payload)
        repaired_payload["updateComponents"] = {
            **payload["updateComponents"],
            "components": [
                {
                    "component": "Text",
                    "id": "root",
                    "text": leaked_value,
                }
            ],
        }
        return repaired_payload

    # Act
    result = validate_outbound_a2ui(invalid_payload, repair=repair_with_secret)

    # Assert
    assert result.valid is False
    assert result.repaired is False
    assert isinstance(result.renderer_part, TextPart)
    diagnostics = result.renderer_part.metadata["developerDiagnostic"]
    exposed_diagnostic_text = repr(
        {
            "validation_errors": result.validation_errors,
            "diagnostics": diagnostics,
            "renderer_part": result.renderer_part.model_dump(mode="json"),
        }
    )
    assert diagnostics["repairAttempted"] is True
    assert leaked_value not in exposed_diagnostic_text
    assert "sk-live-repaired-secret-token-123456789" not in exposed_diagnostic_text
    assert "<redacted-secret>" in exposed_diagnostic_text
    assert "secret-like value" in exposed_diagnostic_text


def test_a2ui_validation_rejects_schema_valid_secret_like_keys() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui

    payload = _a2ui_update(
        components=[
            {
                "component": "Text",
                "id": "root",
                "text": "Treasury services fit the stated need.",
                "OPENROUTER_API_KEY": "configured",
            }
        ]
    )

    # Act
    result = validate_outbound_a2ui(payload)

    # Assert
    assert result.valid is False
    assert isinstance(result.renderer_part, TextPart)
    exposed_diagnostic_text = repr(
        {
            "validation_errors": result.validation_errors,
            "diagnostics": result.renderer_part.metadata["developerDiagnostic"],
        }
    )
    assert "OPENROUTER_API_KEY" not in exposed_diagnostic_text
    assert "<redacted-key>" in exposed_diagnostic_text
    assert "secret-like key" in exposed_diagnostic_text


def test_a2ui_validation_rejects_secret_bearing_data_part_metadata() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui

    leaked_value = "OPENROUTER_API_KEY=sk-live-metadata-secret-token-123456789"
    part = DataPart(
        data=_a2ui_update(),
        metadata={
            "mimeType": A2UI_MIME_TYPE,
            "api_key": leaked_value,
        },
    )

    # Act
    result = validate_outbound_a2ui(part)

    # Assert
    assert result.valid is False
    assert isinstance(result.renderer_part, TextPart)
    exposed_diagnostic_text = repr(
        {
            "validation_errors": result.validation_errors,
            "diagnostics": result.renderer_part.metadata["developerDiagnostic"],
            "renderer_part": result.renderer_part.model_dump(mode="json"),
        }
    )
    assert leaked_value not in exposed_diagnostic_text
    assert "sk-live-metadata-secret-token-123456789" not in exposed_diagnostic_text
    assert "<redacted-secret>" in exposed_diagnostic_text
    assert "<redacted-key>" in exposed_diagnostic_text
    assert "A2UI metadata contains secret-like key" in exposed_diagnostic_text


def test_a2ui_validation_revalidates_existing_data_part_mime_type() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui

    part = DataPart(
        data=_valid_canvas_payload(),
        metadata={"mimeType": A2UI_MIME_TYPE},
    )
    part.metadata["mimeType"] = "text/plain"

    # Act
    result = validate_outbound_a2ui(part)

    # Assert
    assert result.valid is False
    assert isinstance(result.renderer_part, TextPart)
    assert any(
        "metadata.mimeType must be application/json+a2ui" in error
        for error in result.validation_errors
    )


@pytest.mark.parametrize(
    "children",
    [
        [{}],
        [123],
        {"componentId": "component_child"},
        {"path": "items"},
        {"componentId": "component_child", "path": 123},
        {"componentId": "component_child", "path": "items", "extra": True},
    ],
)
def test_a2ui_validation_falls_back_for_invalid_child_references(
    children: object,
) -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui

    invalid_payload = _a2ui_update(
        components=[
            {
                "component": "Column",
                "id": "root",
                "children": children,
            }
        ]
    )

    # Act
    result = validate_outbound_a2ui(invalid_payload)

    # Assert
    assert result.valid is False
    assert isinstance(result.renderer_part, TextPart)
    assert result.validation_errors
    assert (
        result.renderer_part.metadata["developerDiagnostic"]["validationErrors"]
        == result.validation_errors
    )


@pytest.mark.parametrize("action_name", [["approve_plan"], {"type": "approve_plan"}])
def test_a2ui_validation_falls_back_for_non_string_control_action_name(
    action_name: object,
) -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui

    invalid_payload = _valid_canvas_payload()
    button = invalid_payload["updateComponents"]["components"][8]
    button["action"]["event"]["name"] = action_name

    # Act
    result = validate_outbound_a2ui(invalid_payload)

    # Assert
    assert result.valid is False
    assert isinstance(result.renderer_part, TextPart)
    assert result.validation_errors
    assert (
        result.renderer_part.metadata["developerDiagnostic"]["validationErrors"]
        == result.validation_errors
    )
