from datetime import UTC, datetime

import pytest
from pydantic import ValidationError


def test_a2a_message_serializes_required_wire_fields() -> None:
    # Arrange
    from orchestrator_demo.a2a_support.transport import A2AMessage, TextPart

    message = A2AMessage(
        message_id="msg_user_meeting_prep",
        context_id="ctx_abc_manufacturing",
        task_id="task_meeting_prep",
        role="user",
        timestamp=datetime(2026, 5, 30, 12, 0, tzinfo=UTC),
        parts=[TextPart(text="Prepare me for tomorrow's meeting.")],
    )

    # Act
    wire_message = message.model_dump(by_alias=True, mode="json")

    # Assert
    assert wire_message == {
        "messageId": "msg_user_meeting_prep",
        "contextId": "ctx_abc_manufacturing",
        "taskId": "task_meeting_prep",
        "role": "user",
        "timestamp": "2026-05-30T12:00:00Z",
        "parts": [
            {
                "type": "text",
                "text": "Prepare me for tomorrow's meeting.",
            }
        ],
        "metadata": {},
    }


def test_a2a_task_serializes_status_messages_and_context() -> None:
    # Arrange
    from orchestrator_demo.a2a_support.transport import (
        A2AMessage,
        A2ATask,
        TaskStatusUpdate,
        TextPart,
    )

    status = TaskStatusUpdate(
        task_id="task_meeting_prep",
        context_id="ctx_abc_manufacturing",
        status="working",
        timestamp=datetime(2026, 5, 30, 12, 1, tzinfo=UTC),
        message="Plan approval is pending.",
    )
    message = A2AMessage(
        message_id="msg_agent_status",
        context_id="ctx_abc_manufacturing",
        task_id="task_meeting_prep",
        role="agent",
        timestamp=datetime(2026, 5, 30, 12, 1, tzinfo=UTC),
        parts=[TextPart(text="I created a plan for approval.")],
    )
    task = A2ATask(
        task_id="task_meeting_prep",
        context_id="ctx_abc_manufacturing",
        status=status,
        messages=[message],
        metadata={"planId": "plan_meeting_prep"},
    )

    # Act
    wire_task = task.model_dump(by_alias=True, mode="json")

    # Assert
    assert wire_task["taskId"] == "task_meeting_prep"
    assert wire_task["contextId"] == "ctx_abc_manufacturing"
    assert wire_task["status"] == {
        "taskId": "task_meeting_prep",
        "contextId": "ctx_abc_manufacturing",
        "status": "working",
        "timestamp": "2026-05-30T12:01:00Z",
        "message": "Plan approval is pending.",
        "metadata": {},
    }
    assert wire_task["messages"][0]["messageId"] == "msg_agent_status"
    assert wire_task["messages"][0]["role"] == "agent"
    assert wire_task["metadata"] == {"planId": "plan_meeting_prep"}


def test_a2ui_data_part_requires_exact_mime_type() -> None:
    # Arrange
    from orchestrator_demo.a2a_support.transport import A2UI_MIME_TYPE, DataPart

    # Act
    part = DataPart(
        data={"surfaceId": "surface_plan_meeting_prep", "components": []},
        mime_type=A2UI_MIME_TYPE,
    )

    # Assert
    assert part.model_dump(by_alias=True) == {
        "type": "data",
        "mimeType": "application/json+a2ui",
        "data": {"surfaceId": "surface_plan_meeting_prep", "components": []},
        "metadata": {},
    }


def test_a2a_message_accepts_and_serializes_a2ui_data_part_wire_payload() -> None:
    # Arrange
    from orchestrator_demo.a2a_support.transport import (
        A2AMessage,
        A2APart,
        DataPart,
    )

    wire_message = {
        "messageId": "msg_agent_plan_canvas",
        "contextId": "ctx_abc_manufacturing",
        "taskId": "task_meeting_prep",
        "role": "agent",
        "timestamp": "2026-05-30T12:02:00Z",
        "parts": [
            {
                "type": "data",
                "mimeType": "application/json+a2ui",
                "data": {
                    "surfaceId": "surface_plan_meeting_prep",
                    "components": [
                        {
                            "type": "approvalPanel",
                            "planId": "plan_meeting_prep",
                        }
                    ],
                },
                "metadata": {"ownerAgentId": "orchestrator"},
            }
        ],
        "metadata": {"planId": "plan_meeting_prep"},
    }

    # Act
    message = A2AMessage.model_validate(wire_message)
    serialized_message = message.model_dump(by_alias=True, mode="json")

    # Assert
    part = message.parts[0]
    assert isinstance(part, DataPart)
    assert isinstance(part, A2APart)
    assert part.mime_type == "application/json+a2ui"
    assert part.data["surfaceId"] == "surface_plan_meeting_prep"
    assert serialized_message == wire_message


def test_a2ui_data_part_rejects_invalid_mime_type() -> None:
    # Arrange
    from orchestrator_demo.a2a_support.transport import DataPart

    # Act / Assert
    with pytest.raises(ValidationError) as exc_info:
        DataPart.model_validate(
            {
                "type": "data",
                "mimeType": "application/json",
                "data": {
                    "surfaceId": "surface_plan_meeting_prep",
                    "components": [],
                },
            }
        )

    assert "mimeType" in str(exc_info.value)
    assert "application/json+a2ui" in str(exc_info.value)


def test_a2ui_user_action_accepts_plan_action_wire_envelope() -> None:
    # Arrange
    from orchestrator_demo.a2a_support.transport import A2uiUserAction

    # Act
    action = A2uiUserAction.model_validate(
        {
            "userAction": {
                "type": "approve_plan",
                "surfaceId": "surface_plan_meeting_prep",
                "payload": {
                    "planId": "plan_meeting_prep",
                    "approvedStepIds": [
                        "step_internal_notes",
                        "step_synthesis",
                    ],
                    "editedPlanVersion": 2,
                },
            }
        }
    )

    # Assert
    assert action.type == "approve_plan"
    assert action.surface_id == "surface_plan_meeting_prep"
    assert action.plan_id == "plan_meeting_prep"
    assert action.plan_version == 2
    assert action.payload["approvedStepIds"] == [
        "step_internal_notes",
        "step_synthesis",
    ]


def test_a2ui_plan_action_requires_surface_id() -> None:
    # Arrange
    from orchestrator_demo.a2a_support.transport import A2uiUserAction

    # Act / Assert
    with pytest.raises(ValidationError) as exc_info:
        A2uiUserAction.model_validate(
            {
                "userAction": {
                    "type": "approve_plan",
                    "payload": {
                        "planId": "plan_meeting_prep",
                        "planVersion": 1,
                    },
                }
            }
        )

    assert "surfaceId" in str(exc_info.value)


def test_a2ui_plan_action_requires_plan_id_and_version_when_applicable() -> None:
    # Arrange
    from orchestrator_demo.a2a_support.transport import A2uiUserAction

    # Act / Assert
    with pytest.raises(ValidationError) as exc_info:
        A2uiUserAction.model_validate(
            {
                "type": "edit_plan",
                "surfaceId": "surface_plan_meeting_prep",
                "payload": {"instruction": "Focus on treasury products."},
            }
        )

    error_message = str(exc_info.value)
    assert "planId" in error_message
    assert "planVersion" in error_message


def test_part_converters_preserve_a2ui_payload_and_parse_user_action() -> None:
    # Arrange
    from orchestrator_demo.a2a_support.part_converters import (
        a2ui_data_part_from_payload,
        a2ui_user_action_from_part,
        text_part_from_text,
    )
    from orchestrator_demo.a2a_support.transport import A2UI_MIME_TYPE

    payload = {
        "userAction": {
            "type": "reject_plan",
            "surfaceId": "surface_plan_meeting_prep",
            "payload": {
                "planId": "plan_meeting_prep",
                "reason": "Narrow the plan to credit risk only.",
            },
        }
    }

    # Act
    text_part = text_part_from_text("Plan rejected.")
    data_part = a2ui_data_part_from_payload(payload)
    action = a2ui_user_action_from_part(data_part)

    # Assert
    assert text_part.text == "Plan rejected."
    assert data_part.mime_type == A2UI_MIME_TYPE
    assert data_part.data == payload
    assert action.type == "reject_plan"
    assert action.plan_id == "plan_meeting_prep"
