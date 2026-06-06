from __future__ import annotations

import json
import inspect
from typing import Any

import pytest
from google.genai import types

from orchestrator_demo.orchestrator.agent import (
    ORCHESTRATOR_SESSION_STATE_KEY,
    AdkOrchestratorAdapter,
    OrchestratorAgent,
    build_root_agent,
)
from orchestrator_demo.orchestrator.service import (
    OrchestratorService,
    OrchestratorUserActionResult,
)


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


def _assert_draft_updated_contract(
    response: dict[str, Any],
    previous_response: dict[str, Any],
) -> None:
    plan = response["plan"]
    assert response["status"] == "draft_updated"
    assert response["path"] == "draft_updated"
    assert response["planId"] == previous_response["planId"]
    assert response["planId"] == plan["planId"]
    assert response["planVersion"] == previous_response["planVersion"] + 1
    assert response["planVersion"] == plan["planVersion"]
    assert response["approvalSurfaceId"] == previous_response["approvalSurfaceId"]
    assert response["approvalSurfaceId"] == plan["approvalSurfaceId"]
    assert response["stepIds"] == [step["stepId"] for step in plan["steps"]]
    assert response["a2uiParts"]
    assert all(
        part["type"] == "data"
        and part["metadata"]["mimeType"] == "application/json+a2ui"
        for part in response["a2uiParts"]
    )
    assert response["approvalResult"]["graphCreated"] is False
    assert response["approvalResult"]["specialistsCalled"] is False


def _a2ui_update_components(response: dict[str, Any]) -> list[dict[str, Any]]:
    update_parts = [
        part
        for part in response["a2uiParts"]
        if isinstance(part.get("data"), dict)
        and isinstance(part["data"].get("updateComponents"), dict)
    ]
    assert len(update_parts) == 1
    update = update_parts[0]["data"]["updateComponents"]
    assert update["surfaceId"] == response["approvalSurfaceId"]
    assert isinstance(update["components"], list)
    return update["components"]


def _a2ui_component_by_id(
    components: list[dict[str, Any]],
    component_id: str,
) -> dict[str, Any]:
    matches = [component for component in components if component.get("id") == component_id]
    assert len(matches) == 1
    return matches[0]


def _a2ui_step_text(response: dict[str, Any]) -> str:
    components = _a2ui_update_components(response)
    steps = _a2ui_component_by_id(
        components,
        f"component_{response['planId']}_steps",
    )
    return steps["text"]


def _assert_a2ui_update_reflects_plan(response: dict[str, Any]) -> None:
    components = _a2ui_update_components(response)
    metadata = _a2ui_component_by_id(
        components,
        f"component_{response['planId']}_metadata",
    )
    assert f"planId: {response['planId']}" in metadata["text"]
    assert f"planVersion: {response['planVersion']}" in metadata["text"]

    steps_text = _a2ui_step_text(response)
    for index, step in enumerate(response["plan"]["steps"], start=1):
        expected_line = (
            f"{index}. {step['stepId']}: {step['agentId']} - "
            f"{step['instruction']} Expected output: {step['expectedOutput']}"
        )
        assert expected_line in steps_text

    action_payloads = [
        component["action"]["event"]["context"]["payload"]
        for component in components
        if isinstance(component.get("action"), dict)
    ]
    assert action_payloads
    for payload in action_payloads:
        assert payload["planVersion"] == response["planVersion"]
        assert payload["editedPlanVersion"] == response["planVersion"]


def test_root_agent_exposes_required_draft_edit_tools_with_required_fields() -> None:
    # Arrange
    root_agent = build_root_agent(
        adapter=AdkOrchestratorAdapter(),
        model="gemini-2.0-flash",
    )

    # Act
    tool_signatures = {
        tool.name: inspect.signature(tool.func) for tool in root_agent.tools
    }

    # Assert
    assert {
        "add_plan_instruction",
        "remove_plan_step",
        "replace_plan_agent",
        "reorder_plan_steps",
    } <= set(tool_signatures)
    for tool_name in {
        "add_plan_instruction",
        "remove_plan_step",
        "replace_plan_agent",
        "reorder_plan_steps",
    }:
        parameters = tool_signatures[tool_name].parameters
        for required_field in (
            "plan_id",
            "approval_surface_id",
            "edited_plan_version",
        ):
            assert required_field in parameters
            assert parameters[required_field].default is inspect.Parameter.empty

    instruction = root_agent.instruction
    assert "add_plan_instruction" in instruction
    assert "remove_plan_step" in instruction
    assert "replace_plan_agent" in instruction
    assert "reorder_plan_steps" in instruction
    assert "before approve_orchestrator_plan" in instruction


@pytest.mark.asyncio
async def test_draft_edit_tools_synthesize_existing_user_action_envelopes() -> None:
    # Arrange
    class RecordingAgent(OrchestratorAgent):
        def __init__(self) -> None:
            self.user_actions: list[dict[str, Any]] = []

        async def handle_user_action(
            self,
            user_action: Any,
        ) -> OrchestratorUserActionResult:
            self.user_actions.append(user_action)
            return OrchestratorUserActionResult(status="ignored")

    recording_agent = RecordingAgent()
    adapter = AdkOrchestratorAdapter(agent=recording_agent)

    # Act
    await adapter.add_plan_instruction(
        "plan_example",
        "surface_plan_example",
        "step_relationship_summary",
        "Prioritize recent deposit trends.",
        edited_plan_version=7,
    )
    await adapter.remove_plan_step(
        "plan_example",
        "surface_plan_example",
        "step_industry_research",
        edited_plan_version=8,
    )
    await adapter.replace_plan_agent(
        "plan_example",
        "surface_plan_example",
        "step_industry_research",
        "credit_risk",
        edited_plan_version=9,
    )
    await adapter.reorder_plan_steps(
        "plan_example",
        "surface_plan_example",
        ["step_internal_knowledge", "step_relationship_summary"],
        edited_plan_version=10,
    )

    # Assert
    assert recording_agent.user_actions == [
        {
            "userAction": {
                "type": "add_instruction",
                "surfaceId": "surface_plan_example",
                "payload": {
                    "planId": "plan_example",
                    "editedPlanVersion": 7,
                    "stepId": "step_relationship_summary",
                    "instruction": "Prioritize recent deposit trends.",
                },
            }
        },
        {
            "userAction": {
                "type": "remove_step",
                "surfaceId": "surface_plan_example",
                "payload": {
                    "planId": "plan_example",
                    "editedPlanVersion": 8,
                    "stepId": "step_industry_research",
                },
            }
        },
        {
            "userAction": {
                "type": "replace_agent",
                "surfaceId": "surface_plan_example",
                "payload": {
                    "planId": "plan_example",
                    "editedPlanVersion": 9,
                    "stepId": "step_industry_research",
                    "replacementAgentId": "credit_risk",
                },
            }
        },
        {
            "userAction": {
                "type": "reorder_steps",
                "surfaceId": "surface_plan_example",
                "payload": {
                    "planId": "plan_example",
                    "editedPlanVersion": 10,
                    "orderedStepIds": [
                        "step_internal_knowledge",
                        "step_relationship_summary",
                    ],
                },
            }
        },
    ]


@pytest.mark.asyncio
async def test_draft_edit_tools_update_plan_a2ui_and_session_without_execution() -> None:
    # Arrange
    tool_context = FakeToolContext()
    service = OrchestratorService()
    adapter = AdkOrchestratorAdapter(agent=OrchestratorAgent(service))
    submitted = await adapter.submit_orchestrator_request(
        "Prepare me for tomorrow's meeting with ABC Manufacturing.",
        tool_context=tool_context,
    )

    # Act
    instructed = await adapter.add_plan_instruction(
        submitted["planId"],
        submitted["approvalSurfaceId"],
        "step_internal_knowledge",
        "Prioritize covenant follow-up questions.",
        edited_plan_version=submitted["planVersion"],
        tool_context=tool_context,
    )
    removed = await adapter.remove_plan_step(
        instructed["planId"],
        instructed["approvalSurfaceId"],
        "step_industry_research",
        edited_plan_version=instructed["planVersion"],
        tool_context=tool_context,
    )
    replaced = await adapter.replace_plan_agent(
        removed["planId"],
        removed["approvalSurfaceId"],
        "step_relationship_summary",
        "credit_risk",
        edited_plan_version=removed["planVersion"],
        tool_context=tool_context,
    )
    reordered = await adapter.reorder_plan_steps(
        replaced["planId"],
        replaced["approvalSurfaceId"],
        [
            "step_internal_knowledge",
            "step_relationship_summary",
            "step_synthesis",
        ],
        edited_plan_version=replaced["planVersion"],
        tool_context=tool_context,
    )

    # Assert
    for current, previous in (
        (instructed, submitted),
        (removed, instructed),
        (replaced, removed),
        (reordered, replaced),
    ):
        _assert_draft_updated_contract(current, previous)
        _assert_a2ui_update_reflects_plan(current)

    assert "Additional instruction: Prioritize covenant follow-up questions." in (
        _a2ui_step_text(instructed)
    )
    assert "step_industry_research" in _a2ui_step_text(instructed)
    assert "step_industry_research" not in _a2ui_step_text(removed)
    assert "step_relationship_summary: credit_risk" in _a2ui_step_text(replaced)
    assert _a2ui_step_text(reordered).index("1. step_internal_knowledge") < (
        _a2ui_step_text(reordered).index("2. step_relationship_summary")
    )

    assert service.specialist_call_counts() == {}
    assert tool_context.saved_artifacts == []
    snapshot = tool_context.state[ORCHESTRATOR_SESSION_STATE_KEY]
    draft = snapshot["approvalRecords"][submitted["planId"]]["draftPlan"]
    assert draft["plan_version"] == reordered["planVersion"]
    assert [step["step_id"] for step in draft["steps"]] == reordered["stepIds"]
    assert (
        "Additional instruction: Prioritize covenant follow-up questions."
        in reordered["plan"]["steps"][0]["instruction"]
    )
    assert reordered["plan"]["steps"][1]["agentId"] == "credit_risk"


@pytest.mark.asyncio
async def test_failed_draft_edit_preserves_prior_draft_and_returns_safe_error() -> None:
    # Arrange
    tool_context = FakeToolContext()
    adapter = AdkOrchestratorAdapter()
    submitted = await adapter.submit_orchestrator_request(
        "Research this prospect and give me risks, opportunities, and talking points.",
        tool_context=tool_context,
    )
    draft_before_failure = json.loads(
        json.dumps(
            tool_context.state[ORCHESTRATOR_SESSION_STATE_KEY]["approvalRecords"][
                submitted["planId"]
            ]["draftPlan"]
        )
    )
    leaked_replacement_agent = (
        "OPENROUTER_API_KEY=sk-or-v1-replacement-agent-secret-should-not-leak"
    )

    # Act
    failed = await adapter.replace_plan_agent(
        submitted["planId"],
        submitted["approvalSurfaceId"],
        submitted["stepIds"][0],
        leaked_replacement_agent,
        edited_plan_version=submitted["planVersion"],
        tool_context=tool_context,
    )

    # Assert
    assert failed["status"] == "error"
    assert failed["path"] == "error"
    assert failed["error"]["code"] == "invalid_plan_mutation"
    rendered_error = json.dumps(failed, sort_keys=True)
    assert leaked_replacement_agent not in rendered_error
    assert "OPENROUTER_API_KEY" not in rendered_error
    assert "sk-or-v1-replacement-agent-secret-should-not-leak" not in rendered_error
    assert tool_context.saved_artifacts == []
    draft_after_failure = tool_context.state[ORCHESTRATOR_SESSION_STATE_KEY][
        "approvalRecords"
    ][submitted["planId"]]["draftPlan"]
    assert draft_after_failure == draft_before_failure


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
