from __future__ import annotations

import json
from typing import Any

import pytest
from google.genai import types

from orchestrator_demo.orchestrator.agent import (
    ORCHESTRATOR_SESSION_STATE_KEY,
    AdkOrchestratorAdapter,
)


class FakeToolContext:
    def __init__(self) -> None:
        self.state: dict[str, Any] = {}
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


@pytest.mark.asyncio
async def test_direct_request_saves_latest_result_as_text_part_and_records_ref() -> None:
    # Arrange
    tool_context = FakeToolContext()
    adapter = AdkOrchestratorAdapter()

    # Act
    response = await adapter.submit_orchestrator_request(
        "Summarize the internal notes for ABC Manufacturing.",
        tool_context=tool_context,
    )

    # Assert
    assert response["status"] == "direct"
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
async def test_approved_plan_saves_latest_and_plan_execution_text_artifacts() -> None:
    # Arrange
    tool_context = FakeToolContext()
    adapter = AdkOrchestratorAdapter()
    submitted = await adapter.submit_orchestrator_request(
        "Prepare me for tomorrow's meeting with ABC Manufacturing.",
        tool_context=tool_context,
    )
    plan_filename = f"orchestrator_plan_{submitted['planId']}_execution.json"

    # Act
    approved = await adapter.approve_orchestrator_plan(
        submitted["planId"],
        submitted["approvalSurfaceId"],
        submitted["stepIds"],
        edited_plan_version=submitted["planVersion"],
        tool_context=tool_context,
    )

    # Assert
    assert approved["status"] == "approved"
    assert approved["graphCreated"] is True
    assert approved["specialistsCalled"] is True
    assert [artifact["filename"] for artifact in tool_context.saved_artifacts] == [
        "orchestrator_latest_result.json",
        plan_filename,
    ]

    latest_saved, plan_saved = tool_context.saved_artifacts
    latest_document = _artifact_document(latest_saved)
    plan_document = _artifact_document(plan_saved)
    expected_refs = {
        "orchestrator_latest_result.json": {
            "filename": "orchestrator_latest_result.json",
            "version": latest_saved["version"],
            "mimeType": "application/json",
            "documentType": "approved_result",
            "planId": submitted["planId"],
        },
        plan_filename: {
            "filename": plan_filename,
            "version": plan_saved["version"],
            "mimeType": "application/json",
            "documentType": "approved_plan_execution",
            "planId": submitted["planId"],
        },
    }
    assert latest_saved["customMetadata"] == {
        "documentType": "approved_result",
        "mimeType": "application/json",
    }
    assert plan_saved["customMetadata"] == {
        "documentType": "approved_plan_execution",
        "mimeType": "application/json",
    }
    assert latest_document["status"] == "approved"
    assert latest_document["path"] == "approved"
    assert latest_document["planId"] == submitted["planId"]
    assert plan_document["status"] == "approved"
    assert plan_document["planId"] == submitted["planId"]
    assert plan_document["artifacts"]["final_response"]["agent_id"] == "synthesis"
    assert approved["artifactRefs"] == expected_refs
    assert tool_context.state[ORCHESTRATOR_SESSION_STATE_KEY]["artifactRefs"] == (
        expected_refs
    )
