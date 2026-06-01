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
    from a2a.types import Task

    from orchestrator_demo.a2a_support.transport import (
        A2AMessage,
        A2ATask,
        TaskStatusUpdate,
        TextPart,
    )

    status_message = A2AMessage(
        message_id="msg_agent_status",
        context_id="ctx_abc_manufacturing",
        task_id="task_meeting_prep",
        role="agent",
        timestamp=datetime(2026, 5, 30, 12, 1, tzinfo=UTC),
        parts=[TextPart(text="Plan approval is pending.")],
    )
    status = TaskStatusUpdate(
        task_id="task_meeting_prep",
        context_id="ctx_abc_manufacturing",
        status="working",
        timestamp=datetime(2026, 5, 30, 12, 1, tzinfo=UTC),
        message=status_message,
    )
    history_message = A2AMessage(
        message_id="msg_agent_plan",
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
        messages=[history_message],
        metadata={"planId": "plan_meeting_prep"},
    )

    # Act
    wire_task = task.model_dump(by_alias=True, mode="json")
    sdk_task = Task.model_validate(wire_task)

    # Assert
    assert wire_task["id"] == "task_meeting_prep"
    assert "taskId" not in wire_task
    assert wire_task["contextId"] == "ctx_abc_manufacturing"
    assert wire_task["status"] == {
        "state": "working",
        "timestamp": "2026-05-30T12:01:00Z",
        "message": {
            "messageId": "msg_agent_status",
            "contextId": "ctx_abc_manufacturing",
            "taskId": "task_meeting_prep",
            "role": "agent",
            "timestamp": "2026-05-30T12:01:00Z",
            "parts": [
                {
                    "type": "text",
                    "text": "Plan approval is pending.",
                }
            ],
            "metadata": {},
        },
    }
    assert "status" not in wire_task["status"]
    assert "messages" not in wire_task
    assert wire_task["history"][0]["messageId"] == "msg_agent_plan"
    assert wire_task["history"][0]["role"] == "agent"
    assert wire_task["metadata"] == {"planId": "plan_meeting_prep"}
    assert sdk_task.id == "task_meeting_prep"
    assert sdk_task.status.state.value == "working"
    assert sdk_task.status.message is not None
    assert sdk_task.status.message.message_id == "msg_agent_status"


def test_a2ui_data_part_requires_exact_mime_type() -> None:
    # Arrange
    from orchestrator_demo.a2a_support.transport import A2UI_MIME_TYPE, DataPart

    # Act
    part = DataPart(
        data={"surfaceId": "surface_plan_meeting_prep", "components": []},
        metadata={"mimeType": A2UI_MIME_TYPE},
    )

    # Assert
    assert part.model_dump(by_alias=True) == {
        "type": "data",
        "data": {"surfaceId": "surface_plan_meeting_prep", "components": []},
        "metadata": {"mimeType": "application/json+a2ui"},
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
                "data": {
                    "surfaceId": "surface_plan_meeting_prep",
                    "components": [
                        {
                            "type": "approvalPanel",
                            "planId": "plan_meeting_prep",
                        }
                    ],
                },
                "metadata": {
                    "mimeType": "application/json+a2ui",
                    "ownerAgentId": "orchestrator",
                },
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


def test_a2a_message_accepts_sdk_kind_discriminators_for_parts() -> None:
    # Arrange
    from orchestrator_demo.a2a_support.transport import (
        A2AMessage,
        DataPart,
        TextPart,
    )

    wire_message = {
        "messageId": "msg_agent_sdk_parts",
        "contextId": "ctx_abc_manufacturing",
        "taskId": "task_meeting_prep",
        "role": "agent",
        "timestamp": "2026-05-30T12:02:00Z",
        "parts": [
            {
                "kind": "text",
                "text": "I created a plan for approval.",
            },
            {
                "kind": "data",
                "data": {
                    "surfaceId": "surface_plan_meeting_prep",
                    "components": [],
                },
                "metadata": {
                    "mimeType": "application/json+a2ui",
                    "ownerAgentId": "orchestrator",
                },
            },
        ],
        "metadata": {},
    }

    # Act
    message = A2AMessage.model_validate(wire_message)
    serialized_message = message.model_dump(by_alias=True, mode="json")

    # Assert
    assert isinstance(message.parts[0], TextPart)
    assert isinstance(message.parts[1], DataPart)
    assert serialized_message["parts"] == [
        {
            "type": "text",
            "text": "I created a plan for approval.",
        },
        {
            "type": "data",
            "data": {
                "surfaceId": "surface_plan_meeting_prep",
                "components": [],
            },
            "metadata": {
                "mimeType": "application/json+a2ui",
                "ownerAgentId": "orchestrator",
            },
        },
    ]


def test_a2a_message_accepts_sdk_created_a2ui_part_instance() -> None:
    # Arrange
    from a2ui.a2a.parts import create_a2ui_part

    from orchestrator_demo.a2a_support.transport import A2AMessage, DataPart

    sdk_part = create_a2ui_part(
        {
            "surfaceId": "surface_plan_meeting_prep",
            "components": [],
        }
    )
    wire_message = {
        "messageId": "msg_agent_sdk_part_instance",
        "contextId": "ctx_abc_manufacturing",
        "taskId": "task_meeting_prep",
        "role": "agent",
        "timestamp": "2026-05-30T12:02:00Z",
        "parts": [sdk_part],
        "metadata": {},
    }

    # Act
    message = A2AMessage.model_validate(wire_message)
    serialized_message = message.model_dump(by_alias=True, mode="json")

    # Assert
    assert isinstance(message.parts[0], DataPart)
    assert serialized_message["parts"] == [
        {
            "type": "data",
            "data": {
                "surfaceId": "surface_plan_meeting_prep",
                "components": [],
            },
            "metadata": {"mimeType": "application/json+a2ui"},
        }
    ]


def test_a2a_message_accepts_sdk_text_part_with_null_metadata() -> None:
    # Arrange
    from orchestrator_demo.a2a_support.transport import A2AMessage, TextPart

    wire_message = {
        "messageId": "msg_agent_sdk_text",
        "contextId": "ctx_abc_manufacturing",
        "taskId": "task_meeting_prep",
        "role": "agent",
        "timestamp": "2026-05-30T12:02:00Z",
        "parts": [
            {
                "kind": "text",
                "text": "I created a plan for approval.",
                "metadata": None,
            },
        ],
        "metadata": {},
    }

    # Act
    message = A2AMessage.model_validate(wire_message)
    serialized_message = message.model_dump(by_alias=True, mode="json")

    # Assert
    assert isinstance(message.parts[0], TextPart)
    assert serialized_message["parts"] == [
        {
            "type": "text",
            "text": "I created a plan for approval.",
        }
    ]


def test_a2a_message_accepts_sdk_text_part_with_empty_metadata() -> None:
    # Arrange
    from orchestrator_demo.a2a_support.transport import A2AMessage, TextPart

    wire_message = {
        "messageId": "msg_agent_sdk_text_empty_metadata",
        "contextId": "ctx_abc_manufacturing",
        "taskId": "task_meeting_prep",
        "role": "agent",
        "timestamp": "2026-05-30T12:02:00Z",
        "parts": [
            {
                "kind": "text",
                "text": "I created a plan for approval.",
                "metadata": {},
            },
        ],
        "metadata": {},
    }

    # Act
    message = A2AMessage.model_validate(wire_message)
    serialized_message = message.model_dump(by_alias=True, mode="json")

    # Assert
    assert isinstance(message.parts[0], TextPart)
    assert message.parts[0].metadata == {}
    assert serialized_message["parts"] == [
        {
            "type": "text",
            "text": "I created a plan for approval.",
        }
    ]


def test_a2a_message_preserves_sdk_text_part_with_non_empty_metadata() -> None:
    # Arrange
    from orchestrator_demo.a2a_support.transport import A2AMessage, TextPart

    wire_message = {
        "messageId": "msg_agent_sdk_text_metadata",
        "contextId": "ctx_abc_manufacturing",
        "taskId": "task_meeting_prep",
        "role": "agent",
        "timestamp": "2026-05-30T12:02:00Z",
        "parts": [
            {
                "kind": "text",
                "text": "I created a plan for approval.",
                "metadata": {
                    "source": "sdk-client",
                    "sequence": 1,
                },
            },
        ],
        "metadata": {},
    }

    # Act
    message = A2AMessage.model_validate(wire_message)
    serialized_message = message.model_dump(by_alias=True, mode="json")

    # Assert
    assert isinstance(message.parts[0], TextPart)
    assert message.parts[0].metadata == {
        "source": "sdk-client",
        "sequence": 1,
    }
    assert serialized_message["parts"] == [
        {
            "type": "text",
            "text": "I created a plan for approval.",
            "metadata": {
                "source": "sdk-client",
                "sequence": 1,
            },
        }
    ]


def test_a2a_message_accepts_sdk_envelope_with_null_metadata() -> None:
    # Arrange
    from orchestrator_demo.a2a_support.transport import A2AMessage

    wire_message = {
        "messageId": "msg_agent_sdk_envelope",
        "contextId": "ctx_abc_manufacturing",
        "taskId": "task_meeting_prep",
        "role": "agent",
        "timestamp": "2026-05-30T12:02:00Z",
        "parts": [
            {
                "kind": "text",
                "text": "I created a plan for approval.",
            },
        ],
        "metadata": None,
    }

    # Act
    message = A2AMessage.model_validate(wire_message)
    serialized_message = message.model_dump(by_alias=True, mode="json")

    # Assert
    assert message.metadata == {}
    assert serialized_message["metadata"] == {}


def test_task_status_accepts_sdk_envelope_with_null_metadata() -> None:
    # Arrange
    from orchestrator_demo.a2a_support.transport import TaskStatusUpdate

    wire_status = {
        "taskId": "task_meeting_prep",
        "contextId": "ctx_abc_manufacturing",
        "state": "working",
        "timestamp": "2026-05-30T12:03:00Z",
        "metadata": None,
    }

    # Act
    status = TaskStatusUpdate.model_validate(wire_status)
    serialized_status = status.model_dump(by_alias=True, mode="json")

    # Assert
    assert status.metadata == {}
    assert serialized_status["metadata"] == {}
    assert serialized_status["state"] == "working"
    assert "message" not in serialized_status


def test_a2a_task_accepts_sdk_envelope_with_null_metadata() -> None:
    # Arrange
    from orchestrator_demo.a2a_support.transport import A2ATask

    wire_task = {
        "id": "task_meeting_prep",
        "contextId": "ctx_abc_manufacturing",
        "status": {
            "state": "working",
            "timestamp": "2026-05-30T12:03:00Z",
            "metadata": None,
        },
        "history": [],
        "metadata": None,
    }

    # Act
    task = A2ATask.model_validate(wire_task)
    serialized_task = task.model_dump(by_alias=True, mode="json")

    # Assert
    assert task.metadata == {}
    assert task.status.metadata == {}
    assert serialized_task["metadata"] == {}
    assert serialized_task["id"] == "task_meeting_prep"
    assert serialized_task["status"] == {
        "state": "working",
        "timestamp": "2026-05-30T12:03:00Z",
    }
    assert serialized_task["history"] == []


def test_a2a_task_serializes_non_empty_status_metadata() -> None:
    # Arrange
    from orchestrator_demo.a2a_support.transport import A2ATask, TaskStatusUpdate

    status = TaskStatusUpdate(
        task_id="task_meeting_prep",
        context_id="ctx_abc_manufacturing",
        status="auth-required",
        timestamp=datetime(2026, 5, 30, 12, 3, tzinfo=UTC),
        metadata={"retryAfterSeconds": 30, "diagnostic": "reauth required"},
    )
    task = A2ATask(
        task_id="task_meeting_prep",
        context_id="ctx_abc_manufacturing",
        status=status,
    )

    # Act
    serialized_task = task.model_dump(by_alias=True, mode="json")

    # Assert
    assert serialized_task["status"] == {
        "state": "auth-required",
        "timestamp": "2026-05-30T12:03:00Z",
        "metadata": {
            "retryAfterSeconds": 30,
            "diagnostic": "reauth required",
        },
    }


def test_a2a_task_accepts_sdk_task_model_without_sdk_only_fields() -> None:
    # Arrange
    from a2a.types import Task, TaskStatus

    from orchestrator_demo.a2a_support.transport import A2ATask

    sdk_task = Task(
        id="task_meeting_prep",
        contextId="ctx_abc_manufacturing",
        status=TaskStatus(
            state="working",
            timestamp="2026-05-30T12:03:00Z",
        ),
    )

    # Act
    task = A2ATask.model_validate(sdk_task)
    serialized_task = task.model_dump(by_alias=True, mode="json")

    # Assert
    assert task.task_id == "task_meeting_prep"
    assert task.context_id == "ctx_abc_manufacturing"
    assert task.status.task_id == "task_meeting_prep"
    assert task.status.context_id == "ctx_abc_manufacturing"
    assert serialized_task["status"] == {
        "state": "working",
        "timestamp": "2026-05-30T12:03:00Z",
    }
    assert "artifacts" not in serialized_task
    assert "kind" not in serialized_task


def test_a2a_task_preserves_non_empty_sdk_artifacts() -> None:
    # Arrange
    from a2a.types import Artifact, Part, Task, TaskStatus
    from a2a.types import DataPart as SdkDataPart
    from a2a.types import TextPart as SdkTextPart

    from orchestrator_demo.a2a_support.transport import A2ATask, A2UI_MIME_TYPE

    sdk_task = Task(
        id="task_meeting_prep",
        contextId="ctx_abc_manufacturing",
        status=TaskStatus(
            state="completed",
            timestamp="2026-05-30T12:03:00Z",
        ),
        artifacts=[
            Artifact(
                artifactId="artifact_final_output",
                name="Final output",
                parts=[
                    Part(root=SdkTextPart(text="Final brief ready.")),
                    Part(
                        root=SdkDataPart(
                            data={"kind": "text", "text": "A2UI final output"},
                            metadata={"mimeType": A2UI_MIME_TYPE},
                        )
                    ),
                ],
            )
        ],
    )

    # Act
    task = A2ATask.model_validate(sdk_task)
    serialized_task = task.model_dump(by_alias=True, mode="json")

    # Assert
    assert task.artifacts is not None
    assert serialized_task["artifacts"] == [
        {
            "artifactId": "artifact_final_output",
            "name": "Final output",
            "parts": [
                {"type": "text", "text": "Final brief ready."},
                {
                    "type": "data",
                    "data": {"kind": "text", "text": "A2UI final output"},
                    "metadata": {"mimeType": A2UI_MIME_TYPE},
                },
            ],
        }
    ]
    assert "kind" not in serialized_task["artifacts"][0]["parts"][0]


def test_a2a_task_accepts_sdk_task_with_nested_messages() -> None:
    # Arrange
    from a2a.types import Message, Part, Task, TaskStatus
    from a2a.types import TextPart as SdkTextPart

    from orchestrator_demo.a2a_support.transport import A2ATask

    sdk_message = Message(
        messageId="msg_agent_status",
        contextId="ctx_abc_manufacturing",
        taskId="task_meeting_prep",
        role="agent",
        parts=[Part(root=SdkTextPart(text="Plan approval is pending."))],
        extensions=["urn:example:extension"],
        referenceTaskIds=["task_previous"],
    )
    sdk_task = Task(
        id="task_meeting_prep",
        contextId="ctx_abc_manufacturing",
        status=TaskStatus(
            state="working",
            timestamp="2026-05-30T12:03:00Z",
            message=sdk_message,
        ),
        history=[sdk_message],
    )

    # Act
    task = A2ATask.model_validate(sdk_task)
    serialized_task = task.model_dump(by_alias=True, mode="json")

    # Assert
    assert task.status.message is not None
    assert task.status.message.timestamp == datetime(2026, 5, 30, 12, 3, tzinfo=UTC)
    assert task.messages[0].timestamp == datetime(2026, 5, 30, 12, 3, tzinfo=UTC)
    assert serialized_task["status"]["message"] == {
        "messageId": "msg_agent_status",
        "contextId": "ctx_abc_manufacturing",
        "taskId": "task_meeting_prep",
        "role": "agent",
        "timestamp": "2026-05-30T12:03:00Z",
        "parts": [
            {
                "type": "text",
                "text": "Plan approval is pending.",
            }
        ],
        "metadata": {},
    }
    assert serialized_task["history"][0]["timestamp"] == "2026-05-30T12:03:00Z"
    assert "kind" not in serialized_task["status"]["message"]
    assert "extensions" not in serialized_task["status"]["message"]
    assert "referenceTaskIds" not in serialized_task["status"]["message"]


def test_a2a_task_copies_python_name_ids_into_nested_status() -> None:
    # Arrange
    from orchestrator_demo.a2a_support.transport import A2ATask

    task_payload = {
        "task_id": "task_meeting_prep",
        "context_id": "ctx_abc_manufacturing",
        "status": {
            "state": "working",
            "timestamp": "2026-05-30T12:03:00Z",
        },
    }

    # Act
    task = A2ATask.model_validate(task_payload)

    # Assert
    assert task.task_id == "task_meeting_prep"
    assert task.context_id == "ctx_abc_manufacturing"
    assert task.status.task_id == "task_meeting_prep"
    assert task.status.context_id == "ctx_abc_manufacturing"


def test_a2a_task_round_trips_python_name_dump_with_nested_child_ids() -> None:
    # Arrange
    from orchestrator_demo.a2a_support.transport import (
        A2AMessage,
        A2ATask,
        TaskStatusUpdate,
        TextPart,
    )

    status_message = A2AMessage(
        message_id="msg_agent_status",
        context_id="ctx_abc_manufacturing",
        task_id="task_meeting_prep",
        role="agent",
        timestamp=datetime(2026, 5, 30, 12, 3, tzinfo=UTC),
        parts=[TextPart(text="Plan approval is pending.")],
    )
    task = A2ATask(
        task_id="task_meeting_prep",
        context_id="ctx_abc_manufacturing",
        status=TaskStatusUpdate(
            task_id="task_meeting_prep",
            context_id="ctx_abc_manufacturing",
            status="working",
            timestamp=datetime(2026, 5, 30, 12, 3, tzinfo=UTC),
            message=status_message,
        ),
        messages=[status_message],
    )

    # Act
    round_tripped = A2ATask.model_validate(task.model_dump())

    # Assert
    assert round_tripped.task_id == "task_meeting_prep"
    assert round_tripped.status.task_id == "task_meeting_prep"
    assert round_tripped.status.message is not None
    assert round_tripped.status.message.task_id == "task_meeting_prep"
    assert round_tripped.messages[0].context_id == "ctx_abc_manufacturing"


def test_a2a_message_enforces_a2ui_mime_type_for_sdk_kind_data_part() -> None:
    # Arrange
    from orchestrator_demo.a2a_support.transport import A2AMessage

    # Act / Assert
    with pytest.raises(ValidationError) as exc_info:
        A2AMessage.model_validate(
            {
                "messageId": "msg_agent_invalid_sdk_part",
                "contextId": "ctx_abc_manufacturing",
                "taskId": "task_meeting_prep",
                "role": "agent",
                "timestamp": "2026-05-30T12:02:00Z",
                "parts": [
                    {
                        "kind": "data",
                        "data": {
                            "surfaceId": "surface_plan_meeting_prep",
                            "components": [],
                        },
                        "metadata": {"mimeType": "application/json"},
                    }
                ],
                "metadata": {},
            }
        )

    assert "mimeType" in str(exc_info.value)
    assert "application/json+a2ui" in str(exc_info.value)


def test_a2ui_data_part_rejects_invalid_mime_type() -> None:
    # Arrange
    from orchestrator_demo.a2a_support.transport import DataPart

    # Act / Assert
    with pytest.raises(ValidationError) as exc_info:
        DataPart.model_validate(
            {
                "type": "data",
                "data": {
                    "surfaceId": "surface_plan_meeting_prep",
                    "components": [],
                },
                "metadata": {"mimeType": "application/json"},
            }
        )

    assert "mimeType" in str(exc_info.value)
    assert "application/json+a2ui" in str(exc_info.value)


def test_task_status_accepts_and_serializes_a2a_wire_input_and_auth_states() -> None:
    # Arrange
    from orchestrator_demo.a2a_support.transport import TaskStatusUpdate

    wire_states = ("input-required", "auth-required")

    for wire_state in wire_states:
        # Act
        status = TaskStatusUpdate.model_validate(
            {
                "taskId": f"task_{wire_state.replace('-', '_')}",
                "contextId": "ctx_abc_manufacturing",
                "state": wire_state,
                "timestamp": "2026-05-30T12:03:00Z",
            }
        )
        serialized_status = status.model_dump(by_alias=True, mode="json")

        # Assert
        assert status.status == wire_state
        assert serialized_status["state"] == wire_state


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


@pytest.mark.parametrize(
    "action_type",
    [
        "approve_plan",
        "reject_plan",
        "edit_plan",
        "remove_step",
        "reorder_steps",
        "choose_agent",
        "replace_agent",
        "add_instruction",
        "add_instructions",
    ],
)
def test_a2ui_plan_action_names_on_specialist_surface_remain_pass_through(
    action_type: str,
) -> None:
    # Arrange
    from orchestrator_demo.a2a_support.transport import A2uiUserAction

    # Act
    action = A2uiUserAction.model_validate(
        {
            "type": action_type,
            "surfaceId": "surface_product_card",
            "payload": {"cardId": "card_product_opportunities"},
        }
    )

    # Assert
    assert action.type == action_type
    assert action.surface_id == "surface_product_card"
    assert action.plan_id is None
    assert action.plan_version is None
    assert action.payload == {"cardId": "card_product_opportunities"}


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
