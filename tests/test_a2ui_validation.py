from typing import Any

from orchestrator_demo.a2a_support.transport import A2UI_MIME_TYPE, DataPart, TextPart


def _valid_canvas_payload() -> dict[str, Any]:
    return {
        "catalog": "basic",
        "surfaceId": "surface_plan_meeting_prep",
        "planId": "plan_meeting_prep",
        "planVersion": 1,
        "kind": "workflowCanvas",
        "components": [
            {
                "type": "workflowCanvas",
                "id": "component_plan_meeting_prep_canvas",
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
                        "instruction": "Summarize internal notes.",
                        "dependsOn": [],
                        "expectedOutput": "Internal relationship context.",
                        "parallelGroup": "parallel_context",
                    }
                ],
                "parallelGroups": [
                    {
                        "groupId": "parallel_context",
                        "stepIds": ["step_internal_knowledge"],
                    }
                ],
                "controls": [
                    {
                        "controlId": "control_approve_plan",
                        "type": "button",
                        "label": "Approve",
                        "action": {
                            "type": "approve_plan",
                            "surfaceId": "surface_plan_meeting_prep",
                            "planId": "plan_meeting_prep",
                            "planVersion": 1,
                            "payload": {
                                "planId": "plan_meeting_prep",
                                "editedPlanVersion": 1,
                                "approvedStepIds": ["step_internal_knowledge"],
                            },
                        },
                    }
                ],
            }
        ],
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


def test_a2ui_validation_retries_repair_once_and_emits_repaired_data_part() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui

    invalid_payload = _valid_canvas_payload()
    invalid_payload.pop("surfaceId")
    repair_calls: list[tuple[dict[str, Any], list[str]]] = []

    def repair_once(payload: dict[str, Any], errors: list[str]) -> dict[str, Any]:
        repair_calls.append((payload, errors))
        repaired_payload = dict(payload)
        repaired_payload["surfaceId"] = "surface_plan_meeting_prep"
        return repaired_payload

    # Act
    result = validate_outbound_a2ui(invalid_payload, repair=repair_once)

    # Assert
    assert len(repair_calls) == 1
    assert "surfaceId" in repair_calls[0][1][0]
    assert result.valid is True
    assert result.repaired is True
    assert isinstance(result.renderer_part, DataPart)
    assert result.renderer_part.data["surfaceId"] == "surface_plan_meeting_prep"


def test_a2ui_validation_falls_back_to_text_after_failed_repair() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui

    invalid_payload = _valid_canvas_payload()
    invalid_payload["components"] = []
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
