from __future__ import annotations

import json
from typing import Any

import pytest
from google.genai import types

from orchestrator_demo.orchestrator.agent import (
    ORCHESTRATOR_SESSION_STATE_KEY,
    AdkOrchestratorAdapter,
    OrchestratorAgent,
)
from orchestrator_demo.orchestrator.service import OrchestratorService


class FakeActions:
    def __init__(self) -> None:
        self.skip_summarization = False


class FakeToolContext:
    def __init__(self) -> None:
        self.state: dict[str, Any] = {}
        self.actions = FakeActions()
        self.saved_artifacts: list[dict[str, Any]] = []

    async def save_artifact(
        self,
        filename: str,
        artifact: Any,
        custom_metadata: dict[str, Any] | None = None,
    ) -> int:
        version = len(
            [
                saved
                for saved in self.saved_artifacts
                if saved["filename"] == filename
            ]
        )
        self.saved_artifacts.append(
            {
                "filename": filename,
                "artifact": artifact,
                "customMetadata": custom_metadata,
                "version": version,
            }
        )
        return version


def _artifact_document(saved_artifact: dict[str, Any]) -> dict[str, Any]:
    artifact = saved_artifact["artifact"]
    assert isinstance(artifact, types.Part)
    assert artifact.text is not None
    assert artifact.inline_data is None
    return json.loads(artifact.text)


def _assert_plan_response_contract(response: dict[str, Any]) -> None:
    plan = response["plan"]
    assert response["status"] == "plan_required"
    assert response["path"] == "plan_required"
    assert response["planId"] == plan["planId"]
    assert response["planVersion"] == plan["planVersion"]
    assert response["approvalSurfaceId"] == plan["approvalSurfaceId"]
    assert response["stepIds"] == [step["stepId"] for step in plan["steps"]]
    assert response["selectedAgents"] == plan["selectedAgents"]
    assert response["stepInstructions"] == [
        {"stepId": step["stepId"], "instruction": step["instruction"]}
        for step in plan["steps"]
    ]
    assert response["dependencies"] == [
        {"stepId": step["stepId"], "dependsOn": step["dependsOn"]}
        for step in plan["steps"]
    ]
    assert {action["toolName"] for action in response["nextActions"]} == {
        "add_plan_instruction",
        "remove_plan_step",
        "replace_plan_agent",
        "reorder_plan_steps",
        "approve_orchestrator_plan",
        "reject_orchestrator_plan",
    }
    assert response["a2uiParts"]
    assert all(
        part["type"] == "data"
        and part["metadata"]["mimeType"] == "application/json+a2ui"
        for part in response["a2uiParts"]
    )


@pytest.mark.asyncio
async def test_submit_complex_request_restores_session_and_returns_plan_without_specialists() -> None:
    # Arrange
    first_context = FakeToolContext()
    first_service = OrchestratorService()
    first_adapter = AdkOrchestratorAdapter(agent=OrchestratorAgent(first_service))
    first_response = await first_adapter.submit_orchestrator_request(
        "Prepare me for tomorrow's meeting with ABC Manufacturing.",
        tool_context=first_context,
    )
    carried_snapshot = json.loads(
        json.dumps(first_context.state[ORCHESTRATOR_SESSION_STATE_KEY])
    )

    fresh_context = FakeToolContext()
    fresh_context.state[ORCHESTRATOR_SESSION_STATE_KEY] = carried_snapshot
    fresh_service = OrchestratorService()
    fresh_adapter = AdkOrchestratorAdapter(agent=OrchestratorAgent(fresh_service))

    # Act
    second_response = await fresh_adapter.submit_orchestrator_request(
        "Research this prospect and give me risks, opportunities, and talking points.",
        tool_context=fresh_context,
    )

    # Assert
    _assert_plan_response_contract(first_response)
    _assert_plan_response_contract(second_response)
    assert first_service.specialist_call_counts() == {}
    assert fresh_service.specialist_call_counts() == {}
    assert first_context.saved_artifacts == []
    assert fresh_context.saved_artifacts == []
    assert first_context.actions.skip_summarization is True
    assert fresh_context.actions.skip_summarization is True

    snapshot = fresh_context.state[ORCHESTRATOR_SESSION_STATE_KEY]
    assert set(snapshot["approvalRecords"]) == {
        first_response["planId"],
        second_response["planId"],
    }
    assert snapshot["approvalRecords"][first_response["planId"]]["status"] == "draft"
    assert snapshot["approvalRecords"][second_response["planId"]]["status"] == "draft"


@pytest.mark.asyncio
async def test_submit_direct_request_saves_latest_artifact_and_skips_summarization() -> None:
    # Arrange
    tool_context = FakeToolContext()

    # Act
    response = await AdkOrchestratorAdapter().submit_orchestrator_request(
        "Summarize the internal notes for ABC Manufacturing.",
        tool_context=tool_context,
    )

    # Assert
    assert response["status"] == "direct"
    assert response["path"] == "direct"
    assert tool_context.actions.skip_summarization is True
    assert [artifact["filename"] for artifact in tool_context.saved_artifacts] == [
        "orchestrator_latest_result.json"
    ]

    saved = tool_context.saved_artifacts[0]
    document = _artifact_document(saved)
    expected_ref = {
        "filename": "orchestrator_latest_result.json",
        "version": saved["version"],
        "mimeType": "application/json",
        "documentType": "direct_result",
    }
    assert saved["customMetadata"] == {
        "documentType": "direct_result",
        "mimeType": "application/json",
    }
    assert document["status"] == "direct"
    assert document["path"] == "direct"
    assert document["artifacts"]["final_response"]["agent_id"] == (
        response["artifacts"]["final_response"]["agent_id"]
    )
    assert response["artifactRefs"]["orchestrator_latest_result.json"] == expected_ref
    assert tool_context.state[ORCHESTRATOR_SESSION_STATE_KEY]["artifactRefs"][
        "orchestrator_latest_result.json"
    ] == expected_ref


@pytest.mark.asyncio
async def test_submit_error_response_skips_summarization() -> None:
    # Arrange
    tool_context = FakeToolContext()
    tool_context.state[ORCHESTRATOR_SESSION_STATE_KEY] = ["not", "a", "snapshot"]

    # Act
    response = await AdkOrchestratorAdapter().submit_orchestrator_request(
        "Summarize the internal notes for ABC Manufacturing.",
        tool_context=tool_context,
    )

    # Assert
    assert response["status"] == "error"
    assert response["path"] == "error"
    assert response["error"]["code"] == "unexpected_error"
    assert tool_context.actions.skip_summarization is True
    assert tool_context.saved_artifacts == []
