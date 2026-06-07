from __future__ import annotations

import asyncio
import inspect
import json
from types import SimpleNamespace
from typing import Any

import pytest

from orchestrator_demo.agents import build_default_specialists
from orchestrator_demo.orchestrator.agent import (
    ADK_EVAL_MODE_ENV,
    DETERMINISTIC_MODEL_ENV,
    ORCHESTRATOR_SESSION_STATE_KEY,
    AdkOrchestratorAdapter,
    build_root_agent,
)
from orchestrator_demo.orchestrator.response_payloads import (
    ArtifactStorageError,
    build_error_response,
    build_request_response,
    build_user_action_response,
)
from orchestrator_demo.orchestrator.service import (
    OrchestratorService,
    OrchestratorUserActionResult,
)
from orchestrator_demo.orchestrator.surface_routes import (
    SurfaceOwner,
    SurfaceRouteResult,
)


def _approve_event(
    *,
    plan_id: str,
    surface_id: str,
    plan_version: int,
    step_ids: list[str],
) -> dict[str, Any]:
    return {
        "userAction": {
            "type": "approve_plan",
            "surfaceId": surface_id,
            "payload": {
                "planId": plan_id,
                "editedPlanVersion": plan_version,
                "approvedStepIds": step_ids,
            },
        }
    }


def _add_instruction_event(
    *,
    plan_id: str,
    surface_id: str,
    plan_version: int,
    step_id: str,
) -> dict[str, Any]:
    return {
        "userAction": {
            "type": "add_instruction",
            "surfaceId": surface_id,
            "payload": {
                "planId": plan_id,
                "editedPlanVersion": plan_version,
                "stepId": step_id,
                "instruction": "Prioritize recent deposit trends.",
            },
        }
    }


def _reject_event(
    *,
    plan_id: str,
    surface_id: str,
    plan_version: int,
) -> dict[str, Any]:
    return {
        "userAction": {
            "type": "reject_plan",
            "surfaceId": surface_id,
            "payload": {
                "planId": plan_id,
                "editedPlanVersion": plan_version,
                "reason": "Too broad for this meeting.",
            },
        }
    }


def _assert_json_dict(payload: dict[str, Any]) -> None:
    assert isinstance(payload, dict)
    json.dumps(payload, sort_keys=True)


def _assert_text_absent(rendered: str, forbidden: str, label: str) -> None:
    if forbidden in rendered:
        pytest.fail(f"payload leaked {label}")


def _assert_plan_contract(payload: dict[str, Any]) -> None:
    plan = payload["plan"]
    assert payload["planId"] == plan["planId"]
    assert payload["planVersion"] == plan["planVersion"]
    assert payload["approvalSurfaceId"] == plan["approvalSurfaceId"]
    assert payload["selectedAgents"] == plan["selectedAgents"]
    assert payload["stepIds"] == [step["stepId"] for step in plan["steps"]]
    assert payload["dependencies"] == [
        {"stepId": step["stepId"], "dependsOn": step["dependsOn"]}
        for step in plan["steps"]
    ]
    assert payload["riskNotes"] == plan["riskNotes"]
    assert {action["toolName"] for action in payload["nextActions"]} == {
        "add_plan_instruction",
        "remove_plan_step",
        "replace_plan_agent",
        "reorder_plan_steps",
        "approve_orchestrator_plan",
        "reject_orchestrator_plan",
    }
    assert payload["a2uiParts"]
    assert all(
        part["type"] == "data"
        and part["metadata"]["mimeType"] == "application/json+a2ui"
        for part in payload["a2uiParts"]
    )


class FakeToolContext:
    def __init__(
        self,
        *,
        app_name: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        self.state: dict[str, Any] = {}
        self.saved_artifacts: list[dict[str, Any]] = []
        if session_id is not None:
            self.session = SimpleNamespace(
                app_name=app_name or "orchestrator",
                user_id=user_id or "relationship_manager",
                id=session_id,
            )
            self.user_id = user_id or "relationship_manager"

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


class YieldingSaveToolContext(FakeToolContext):
    def __init__(self, save_started: asyncio.Event) -> None:
        super().__init__()
        self._save_started = save_started

    async def save_artifact(
        self,
        filename: str,
        artifact: Any,
        custom_metadata: dict[str, Any] | None = None,
    ) -> int:
        self._save_started.set()
        await asyncio.sleep(0.25)
        return await super().save_artifact(filename, artifact, custom_metadata)


class FailingSaveToolContext(FakeToolContext):
    async def save_artifact(
        self,
        filename: str,
        artifact: Any,
        custom_metadata: dict[str, Any] | None = None,
    ) -> int:
        raise OSError("disk full sk-or-v1-artifact-save-secret")


def _saved_artifact_json(saved_artifact: dict[str, Any]) -> dict[str, Any]:
    artifact = saved_artifact["artifact"]
    assert artifact.text is not None
    assert artifact.inline_data is None
    return json.loads(artifact.text)


def test_root_instruction_includes_plan_version_for_followup_tools() -> None:
    # Arrange
    agent = build_root_agent(
        adapter=AdkOrchestratorAdapter(),
        model="gemini-2.0-flash",
    )

    # Assert
    assert "planVersion" in agent.instruction
    assert "approvalSurfaceId" in agent.instruction
    assert "current planVersion" in agent.instruction


@pytest.mark.asyncio
async def test_adk_adapter_registers_advertised_next_action_tools() -> None:
    # Arrange
    adapter = AdkOrchestratorAdapter()
    submitted = await adapter.submit_orchestrator_request(
        "Prepare me for tomorrow's meeting with ABC Manufacturing."
    )

    # Act
    updated = await adapter.add_plan_instruction(
        submitted["planId"],
        submitted["approvalSurfaceId"],
        submitted["stepIds"][0],
        "Prioritize recent deposit trends.",
        edited_plan_version=submitted["planVersion"],
    )

    # Assert
    assert updated["status"] == "draft_updated"
    assert updated["planVersion"] == submitted["planVersion"] + 1

    tool_signatures = {
        tool.name: inspect.signature(tool.func)
        for tool in adapter.tools()
    }
    tool_parameters = {
        tool_name: set(signature.parameters)
        for tool_name, signature in tool_signatures.items()
    }
    for payload in (submitted, updated):
        assert {action["toolName"] for action in payload["nextActions"]} <= set(
            tool_parameters
        )
        for action in payload["nextActions"]:
            assert set(action["requiredFields"]) <= tool_parameters[action["toolName"]]

    assert "approval_surface_id" in tool_parameters["approve_orchestrator_plan"]
    assert "approval_surface_id" in tool_parameters["reject_orchestrator_plan"]
    assert "surface_id" not in tool_parameters["approve_orchestrator_plan"]
    assert "surface_id" not in tool_parameters["reject_orchestrator_plan"]
    for tool_name in {
        "add_plan_instruction",
        "remove_plan_step",
        "replace_plan_agent",
        "reorder_plan_steps",
        "approve_orchestrator_plan",
    }:
        assert (
            tool_signatures[tool_name]
            .parameters["edited_plan_version"]
            .default
            is inspect.Parameter.empty
        )
    assert (
        tool_signatures["reject_orchestrator_plan"]
        .parameters["edited_plan_version"]
        .default
        is None
    )
    assert (
        tool_signatures["approve_orchestrator_plan"]
        .parameters["tool_context"]
        .default
        is inspect.Parameter.empty
    )
    assert (
        tool_signatures["reject_orchestrator_plan"]
        .parameters["tool_context"]
        .default
        is inspect.Parameter.empty
    )


@pytest.mark.asyncio
async def test_adk_adapter_hydrates_plan_actions_from_tool_context_state() -> None:
    # Arrange
    tool_context = FakeToolContext()
    submitted = await AdkOrchestratorAdapter().submit_orchestrator_request(
        "Prepare me for tomorrow's meeting with ABC Manufacturing.",
        tool_context=tool_context,
    )

    # Act
    updated = await AdkOrchestratorAdapter().add_plan_instruction(
        submitted["planId"],
        submitted["approvalSurfaceId"],
        submitted["stepIds"][0],
        "Prioritize recent deposit trends.",
        edited_plan_version=submitted["planVersion"],
        tool_context=tool_context,
    )
    approved = await AdkOrchestratorAdapter().approve_orchestrator_plan(
        updated["planId"],
        updated["approvalSurfaceId"],
        updated["stepIds"],
        edited_plan_version=updated["planVersion"],
        tool_context=tool_context,
    )

    # Assert
    assert ORCHESTRATOR_SESSION_STATE_KEY in tool_context.state
    assert submitted["status"] == "plan_required"
    assert updated["status"] == "draft_updated"
    assert updated["planVersion"] == submitted["planVersion"] + 1
    assert approved["status"] == "approved"
    assert approved["approvalResult"]["graphCreated"] is True
    assert approved["approvalResult"]["specialistsCalled"] is True


@pytest.mark.asyncio
async def test_adk_adapter_resets_shared_state_for_fresh_tool_context() -> None:
    # Arrange
    adapter = AdkOrchestratorAdapter()
    first_context = FakeToolContext()
    fresh_context = FakeToolContext()
    submitted = await adapter.submit_orchestrator_request(
        "Prepare me for tomorrow's meeting with ABC Manufacturing.",
        tool_context=first_context,
    )

    # Act
    blocked = await adapter.approve_orchestrator_plan(
        submitted["planId"],
        submitted["approvalSurfaceId"],
        submitted["stepIds"],
        edited_plan_version=submitted["planVersion"],
        tool_context=fresh_context,
    )

    # Assert
    _assert_json_dict(blocked)
    assert blocked["status"] == "error"
    assert blocked["error"]["routeCode"] == "unknown_surface"
    fresh_snapshot = fresh_context.state[ORCHESTRATOR_SESSION_STATE_KEY]
    assert fresh_snapshot["approvalRecords"] == {}
    assert fresh_snapshot["requestContextsByPlanId"] == {}
    assert fresh_snapshot["surfaceRegistry"]["ownersBySurfaceId"] == {}


@pytest.mark.asyncio
async def test_adk_adapter_persists_direct_route_artifact_to_tool_context() -> None:
    # Arrange
    tool_context = FakeToolContext()

    # Act
    response = await AdkOrchestratorAdapter().submit_orchestrator_request(
        "Summarize the internal notes for ABC Manufacturing.",
        tool_context=tool_context,
    )

    # Assert
    assert response["status"] == "direct"
    assert [artifact["filename"] for artifact in tool_context.saved_artifacts] == [
        "orchestrator_latest_result.json"
    ]
    saved = tool_context.saved_artifacts[0]
    document = _saved_artifact_json(saved)
    artifact_ref = {
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
    assert response["artifactRefs"]["orchestrator_latest_result.json"] == artifact_ref
    assert tool_context.state[ORCHESTRATOR_SESSION_STATE_KEY]["artifactRefs"][
        "orchestrator_latest_result.json"
    ] == artifact_ref


@pytest.mark.asyncio
async def test_adk_adapter_returns_direct_result_when_artifact_save_fails() -> None:
    # Arrange
    leaked_secret = "sk-or-v1-artifact-save-secret"
    tool_context = FailingSaveToolContext()

    # Act
    response = await AdkOrchestratorAdapter().submit_orchestrator_request(
        "Summarize the internal notes for ABC Manufacturing.",
        tool_context=tool_context,
    )

    # Assert
    _assert_json_dict(response)
    assert response["status"] == "direct"
    assert response["path"] == "direct"
    assert response["artifacts"]["final_response"]["agent_id"] == (
        "internal_knowledge"
    )
    assert response["artifactRefs"] == {}
    assert response["artifactPersistence"]["status"] == "failed"
    assert response["artifactPersistence"]["error"]["code"] == (
        "artifact_storage_error"
    )
    assert tool_context.saved_artifacts == []
    assert tool_context.state[ORCHESTRATOR_SESSION_STATE_KEY]["artifactRefs"] == {}
    rendered = json.dumps(response)
    assert leaked_secret not in rendered
    assert "Traceback" not in rendered


@pytest.mark.asyncio
async def test_adk_adapter_persists_approved_graph_artifact_to_tool_context() -> None:
    # Arrange
    tool_context = FakeToolContext()
    submitted = await AdkOrchestratorAdapter().submit_orchestrator_request(
        "Prepare me for tomorrow's meeting with ABC Manufacturing.",
        tool_context=tool_context,
    )

    # Act
    approved = await AdkOrchestratorAdapter().approve_orchestrator_plan(
        submitted["planId"],
        submitted["approvalSurfaceId"],
        submitted["stepIds"],
        edited_plan_version=submitted["planVersion"],
        tool_context=tool_context,
    )

    # Assert
    latest_filename = "orchestrator_latest_result.json"
    filename = f"orchestrator_plan_{submitted['planId']}_execution.json"
    assert approved["status"] == "approved"
    assert [artifact["filename"] for artifact in tool_context.saved_artifacts] == [
        latest_filename,
        filename
    ]
    latest_saved = tool_context.saved_artifacts[0]
    saved = tool_context.saved_artifacts[1]
    document = _saved_artifact_json(saved)
    latest_artifact_ref = {
        "filename": latest_filename,
        "version": latest_saved["version"],
        "mimeType": "application/json",
        "documentType": "approved_result",
        "planId": submitted["planId"],
    }
    artifact_ref = {
        "filename": filename,
        "version": saved["version"],
        "mimeType": "application/json",
        "documentType": "approved_plan_execution",
        "planId": submitted["planId"],
    }
    assert latest_saved["customMetadata"] == {
        "documentType": "approved_result",
        "mimeType": "application/json",
    }
    assert saved["customMetadata"] == {
        "documentType": "approved_plan_execution",
        "mimeType": "application/json",
    }
    assert document["status"] == "approved"
    assert document["planId"] == submitted["planId"]
    assert document["artifacts"]["final_response"]["agent_id"] == "synthesis"
    assert approved["artifactRefs"][latest_filename] == latest_artifact_ref
    assert approved["artifactRefs"][filename] == artifact_ref
    assert tool_context.state[ORCHESTRATOR_SESSION_STATE_KEY]["artifactRefs"][
        latest_filename
    ] == latest_artifact_ref
    assert tool_context.state[ORCHESTRATOR_SESSION_STATE_KEY]["artifactRefs"][
        filename
    ] == artifact_ref


@pytest.mark.asyncio
async def test_adk_adapter_returns_approved_result_when_artifact_save_fails() -> None:
    # Arrange
    leaked_secret = "sk-or-v1-artifact-save-secret"
    tool_context = FailingSaveToolContext()
    submitted = await AdkOrchestratorAdapter().submit_orchestrator_request(
        "Prepare me for tomorrow's meeting with ABC Manufacturing.",
        tool_context=tool_context,
    )

    # Act
    approved = await AdkOrchestratorAdapter().approve_orchestrator_plan(
        submitted["planId"],
        submitted["approvalSurfaceId"],
        submitted["stepIds"],
        edited_plan_version=submitted["planVersion"],
        tool_context=tool_context,
    )

    # Assert
    _assert_json_dict(approved)
    assert approved["status"] == "approved"
    assert approved["approvalResult"]["graphCreated"] is True
    assert approved["approvalResult"]["specialistsCalled"] is True
    assert approved["artifacts"]["final_response"]["agent_id"] == "synthesis"
    assert approved["statusEvents"]
    assert approved["a2uiParts"]
    assert approved["artifactRefs"] == {}
    assert approved["artifactPersistence"]["status"] == "failed"
    assert approved["artifactPersistence"]["error"]["code"] == "artifact_storage_error"
    assert tool_context.saved_artifacts == []
    snapshot = tool_context.state[ORCHESTRATOR_SESSION_STATE_KEY]
    assert snapshot["approvalRecords"][submitted["planId"]]["status"] == "approved"
    assert snapshot["artifactRefs"] == {}
    rendered = json.dumps(approved)
    assert leaked_secret not in rendered
    assert "Traceback" not in rendered


@pytest.mark.asyncio
async def test_adk_adapter_serializes_concurrent_session_restore_execute_persist() -> None:
    # Arrange
    adapter = AdkOrchestratorAdapter()
    first_save_started = asyncio.Event()
    first_context = YieldingSaveToolContext(first_save_started)
    second_context = FakeToolContext()

    first_submitted = await adapter.submit_orchestrator_request(
        "Prepare me for tomorrow's meeting with ABC Manufacturing.",
        tool_context=first_context,
    )
    second_submitted = await adapter.submit_orchestrator_request(
        "Research this prospect and give me risks, opportunities, and talking points.",
        tool_context=second_context,
    )

    # Act
    first_task = asyncio.create_task(
        adapter.approve_orchestrator_plan(
            first_submitted["planId"],
            first_submitted["approvalSurfaceId"],
            first_submitted["stepIds"],
            edited_plan_version=first_submitted["planVersion"],
            tool_context=first_context,
        )
    )
    await asyncio.wait_for(first_save_started.wait(), timeout=5)
    second_task = asyncio.create_task(
        adapter.approve_orchestrator_plan(
            second_submitted["planId"],
            second_submitted["approvalSurfaceId"],
            second_submitted["stepIds"],
            edited_plan_version=second_submitted["planVersion"],
            tool_context=second_context,
        )
    )
    first_approved, second_approved = await asyncio.gather(first_task, second_task)

    # Assert
    first_plan_id = first_submitted["planId"]
    second_plan_id = second_submitted["planId"]
    latest_filename = "orchestrator_latest_result.json"
    first_filename = f"orchestrator_plan_{first_plan_id}_execution.json"
    second_filename = f"orchestrator_plan_{second_plan_id}_execution.json"
    first_snapshot = first_context.state[ORCHESTRATOR_SESSION_STATE_KEY]
    second_snapshot = second_context.state[ORCHESTRATOR_SESSION_STATE_KEY]

    assert first_approved["status"] == "approved"
    assert second_approved["status"] == "approved"
    assert set(first_snapshot["approvalRecords"]) == {first_plan_id}
    assert set(second_snapshot["approvalRecords"]) == {second_plan_id}
    assert first_snapshot["approvalRecords"][first_plan_id]["status"] == "approved"
    assert second_snapshot["approvalRecords"][second_plan_id]["status"] == "approved"
    assert set(first_snapshot["requestContextsByPlanId"]) == {first_plan_id}
    assert set(second_snapshot["requestContextsByPlanId"]) == {second_plan_id}
    assert set(first_snapshot["artifactRefs"]) == {latest_filename, first_filename}
    assert set(second_snapshot["artifactRefs"]) == {latest_filename, second_filename}
    assert set(first_approved["artifactRefs"]) == {latest_filename, first_filename}
    assert set(second_approved["artifactRefs"]) == {latest_filename, second_filename}
    assert [artifact["filename"] for artifact in first_context.saved_artifacts] == [
        latest_filename,
        first_filename
    ]
    assert [artifact["filename"] for artifact in second_context.saved_artifacts] == [
        latest_filename,
        second_filename
    ]


@pytest.mark.asyncio
async def test_adk_adapter_preserves_finalized_state_for_stale_approval_snapshot() -> None:
    # Arrange
    adapter = AdkOrchestratorAdapter()
    live_context = FakeToolContext()
    submitted = await adapter.submit_orchestrator_request(
        "Prepare me for tomorrow's meeting with ABC Manufacturing.",
        tool_context=live_context,
    )
    stale_context = FakeToolContext()
    stale_context.state[ORCHESTRATOR_SESSION_STATE_KEY] = json.loads(
        json.dumps(live_context.state[ORCHESTRATOR_SESSION_STATE_KEY])
    )

    # Act
    approved = await adapter.approve_orchestrator_plan(
        submitted["planId"],
        submitted["approvalSurfaceId"],
        submitted["stepIds"],
        edited_plan_version=submitted["planVersion"],
        tool_context=live_context,
    )
    replayed = await adapter.approve_orchestrator_plan(
        submitted["planId"],
        submitted["approvalSurfaceId"],
        submitted["stepIds"],
        edited_plan_version=submitted["planVersion"],
        tool_context=stale_context,
    )

    # Assert
    latest_filename = "orchestrator_latest_result.json"
    filename = f"orchestrator_plan_{submitted['planId']}_execution.json"
    replayed_snapshot = stale_context.state[ORCHESTRATOR_SESSION_STATE_KEY]

    assert approved["status"] == "approved"
    assert replayed["status"] == "error"
    assert replayed["error"]["code"] == "plan_already_final"
    assert [artifact["filename"] for artifact in live_context.saved_artifacts] == [
        latest_filename,
        filename
    ]
    assert stale_context.saved_artifacts == []
    assert replayed_snapshot["approvalRecords"][submitted["planId"]]["status"] == (
        "approved"
    )
    assert replayed_snapshot["requestContextsByPlanId"][submitted["planId"]][
        "approvedPlanId"
    ] == submitted["planId"]
    assert set(replayed_snapshot["artifactRefs"]) == {latest_filename, filename}


@pytest.mark.asyncio
async def test_adk_adapter_preserves_newer_draft_for_stale_plan_actions() -> None:
    # Arrange
    adapter = AdkOrchestratorAdapter()
    live_context = FakeToolContext()
    submitted = await adapter.submit_orchestrator_request(
        "Prepare me for tomorrow's meeting with ABC Manufacturing.",
        tool_context=live_context,
    )
    stale_contexts = [FakeToolContext(), FakeToolContext(), FakeToolContext()]
    for stale_context in stale_contexts:
        stale_context.state[ORCHESTRATOR_SESSION_STATE_KEY] = json.loads(
            json.dumps(live_context.state[ORCHESTRATOR_SESSION_STATE_KEY])
        )
    original_version = submitted["planVersion"]

    updated = await adapter.add_plan_instruction(
        submitted["planId"],
        submitted["approvalSurfaceId"],
        submitted["stepIds"][0],
        "Prioritize recent deposit trends.",
        edited_plan_version=original_version,
        tool_context=live_context,
    )

    # Act
    stale_edit = await adapter.add_plan_instruction(
        submitted["planId"],
        submitted["approvalSurfaceId"],
        submitted["stepIds"][0],
        "This stale edit must not be applied.",
        edited_plan_version=original_version,
        tool_context=stale_contexts[0],
    )
    stale_approval = await adapter.approve_orchestrator_plan(
        submitted["planId"],
        submitted["approvalSurfaceId"],
        submitted["stepIds"],
        edited_plan_version=original_version,
        tool_context=stale_contexts[1],
    )
    stale_rejection = await adapter.reject_orchestrator_plan(
        submitted["planId"],
        submitted["approvalSurfaceId"],
        "Rejecting an outdated draft.",
        edited_plan_version=original_version,
        tool_context=stale_contexts[2],
    )

    # Assert
    assert updated["status"] == "draft_updated"
    assert updated["planVersion"] == original_version + 1
    for response in (stale_edit, stale_approval, stale_rejection):
        _assert_json_dict(response)
        assert response["status"] == "error"
        assert response["error"]["code"] == "stale_plan_version"

    for stale_context in stale_contexts:
        snapshot = stale_context.state[ORCHESTRATOR_SESSION_STATE_KEY]
        record = snapshot["approvalRecords"][submitted["planId"]]
        assert record["status"] == "draft"
        assert record["draftPlan"]["plan_version"] == updated["planVersion"]
        rendered_snapshot = json.dumps(snapshot)
        assert "This stale edit must not be applied." not in rendered_snapshot

    assert all(stale_context.saved_artifacts == [] for stale_context in stale_contexts)


@pytest.mark.asyncio
async def test_adk_adapter_scopes_deterministic_plan_cache_by_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    monkeypatch.setenv(DETERMINISTIC_MODEL_ENV, "1")
    monkeypatch.setenv(ADK_EVAL_MODE_ENV, "1")
    adapter = AdkOrchestratorAdapter()
    first_context = FakeToolContext(session_id="session_a")
    second_context = FakeToolContext(session_id="session_b")
    user_input = "Prepare me for tomorrow's meeting with ABC Manufacturing."

    first_submitted = await adapter.submit_orchestrator_request(
        user_input,
        tool_context=first_context,
    )
    second_submitted = await adapter.submit_orchestrator_request(
        user_input,
        tool_context=second_context,
    )

    # Act
    first_updated = await adapter.add_plan_instruction(
        first_submitted["planId"],
        first_submitted["approvalSurfaceId"],
        first_submitted["stepIds"][0],
        "Prioritize recent deposit trends.",
        edited_plan_version=first_submitted["planVersion"],
        tool_context=first_context,
    )
    second_rejected = await adapter.reject_orchestrator_plan(
        second_submitted["planId"],
        second_submitted["approvalSurfaceId"],
        "Not needed for this synthetic session.",
        edited_plan_version=second_submitted["planVersion"],
        tool_context=second_context,
    )

    # Assert
    assert first_submitted["planId"] == second_submitted["planId"]
    assert first_updated["status"] == "draft_updated"
    assert first_updated["planVersion"] == first_submitted["planVersion"] + 1
    assert second_rejected["status"] == "rejected"

    first_snapshot = first_context.state[ORCHESTRATOR_SESSION_STATE_KEY]
    second_snapshot = second_context.state[ORCHESTRATOR_SESSION_STATE_KEY]
    first_record = first_snapshot["approvalRecords"][first_submitted["planId"]]
    second_record = second_snapshot["approvalRecords"][second_submitted["planId"]]
    assert first_record["status"] == "draft"
    assert first_record["draftPlan"]["plan_version"] == first_updated["planVersion"]
    assert second_record["status"] == "rejected"
    assert second_record["draftPlan"]["plan_version"] == second_submitted["planVersion"]


@pytest.mark.asyncio
async def test_adk_adapter_accepts_fresh_deterministic_draft_after_empty_state_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    monkeypatch.setenv(DETERMINISTIC_MODEL_ENV, "1")
    monkeypatch.setenv(ADK_EVAL_MODE_ENV, "1")
    adapter = AdkOrchestratorAdapter()
    tool_context = FakeToolContext(session_id="session_reset")
    user_input = "Prepare me for tomorrow's meeting with ABC Manufacturing."

    first_submitted = await adapter.submit_orchestrator_request(
        user_input,
        tool_context=tool_context,
    )
    first_rejected = await adapter.reject_orchestrator_plan(
        first_submitted["planId"],
        first_submitted["approvalSurfaceId"],
        "Not needed for this synthetic session.",
        edited_plan_version=first_submitted["planVersion"],
        tool_context=tool_context,
    )

    # Simulate a fresh ADK eval/session turn reusing the same session identity.
    tool_context.state.clear()

    # Act
    second_submitted = await adapter.submit_orchestrator_request(
        user_input,
        tool_context=tool_context,
    )
    second_rejected = await adapter.reject_orchestrator_plan(
        second_submitted["planId"],
        second_submitted["approvalSurfaceId"],
        "Still not needed for this synthetic session.",
        edited_plan_version=second_submitted["planVersion"],
        tool_context=tool_context,
    )

    # Assert
    assert first_rejected["status"] == "rejected"
    assert second_submitted["status"] == "plan_required"
    assert second_submitted["planId"] == first_submitted["planId"]
    assert second_rejected["status"] == "rejected"
    assert second_rejected["graphCreated"] is False
    assert second_rejected["specialistsCalled"] is False

    snapshot = tool_context.state[ORCHESTRATOR_SESSION_STATE_KEY]
    record = snapshot["approvalRecords"][second_submitted["planId"]]
    assert record["status"] == "rejected"
    assert record["draftPlan"]["plan_version"] == second_submitted["planVersion"]


@pytest.mark.asyncio
async def test_adk_adapter_clears_edited_draft_cache_after_empty_state_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    monkeypatch.setenv(DETERMINISTIC_MODEL_ENV, "1")
    monkeypatch.setenv(ADK_EVAL_MODE_ENV, "1")
    adapter = AdkOrchestratorAdapter()
    tool_context = FakeToolContext(session_id="session_edited_reset")
    user_input = "Prepare me for tomorrow's meeting with ABC Manufacturing."

    first_submitted = await adapter.submit_orchestrator_request(
        user_input,
        tool_context=tool_context,
    )
    first_updated = await adapter.add_plan_instruction(
        first_submitted["planId"],
        first_submitted["approvalSurfaceId"],
        first_submitted["stepIds"][0],
        "Prioritize recent deposit trends.",
        edited_plan_version=first_submitted["planVersion"],
        tool_context=tool_context,
    )

    # Simulate a fresh ADK eval/session turn reusing the same session identity.
    tool_context.state.clear()

    # Act
    second_submitted = await adapter.submit_orchestrator_request(
        user_input,
        tool_context=tool_context,
    )
    second_rejected = await adapter.reject_orchestrator_plan(
        second_submitted["planId"],
        second_submitted["approvalSurfaceId"],
        "Do not run this synthetic workflow.",
        edited_plan_version=second_submitted["planVersion"],
        tool_context=tool_context,
    )

    # Assert
    assert first_updated["status"] == "draft_updated"
    assert first_updated["planVersion"] == first_submitted["planVersion"] + 1
    assert second_submitted["status"] == "plan_required"
    assert second_submitted["planId"] == first_submitted["planId"]
    assert second_submitted["planVersion"] == first_submitted["planVersion"]
    assert second_rejected["status"] == "rejected"
    assert second_rejected["planVersion"] == second_submitted["planVersion"]

    snapshot = tool_context.state[ORCHESTRATOR_SESSION_STATE_KEY]
    record = snapshot["approvalRecords"][second_submitted["planId"]]
    assert record["status"] == "rejected"
    assert record["draftPlan"]["plan_version"] == second_submitted["planVersion"]


@pytest.mark.asyncio
async def test_response_builders_cover_success_statuses_with_json_contract() -> None:
    # Arrange
    specialists = build_default_specialists()
    direct_service = OrchestratorService()
    clarification_service = OrchestratorService(
        specialists={"internal_knowledge": specialists["internal_knowledge"]}
    )
    plan_service = OrchestratorService()
    rejection_service = OrchestratorService()

    # Act
    direct = build_request_response(
        await direct_service.handle_user_request(
            "Summarize the internal notes for ABC Manufacturing."
        )
    )
    clarification = build_request_response(
        await clarification_service.handle_user_request(
            "Prepare me for tomorrow's meeting with ABC Manufacturing."
        )
    )
    proposed = await plan_service.handle_user_request(
        "Prepare me for tomorrow's meeting with ABC Manufacturing."
    )
    plan_required = build_request_response(proposed)
    assert proposed.approval_plan is not None
    edited_result = await plan_service.handle_user_action(
        _add_instruction_event(
            plan_id=proposed.approval_plan.plan_id,
            surface_id=proposed.approval_plan.approval_surface_id or "",
            plan_version=proposed.approval_plan.plan_version,
            step_id=proposed.approval_plan.steps[0].step_id,
        )
    )
    draft_updated = build_user_action_response(edited_result)
    updated_plan = edited_result.approval_result.draft_plan
    assert updated_plan is not None
    approved = build_user_action_response(
        await plan_service.handle_user_action(
            _approve_event(
                plan_id=updated_plan.plan_id,
                surface_id=updated_plan.approval_surface_id or "",
                plan_version=updated_plan.plan_version,
                step_ids=[step.step_id for step in updated_plan.steps],
            )
        )
    )
    rejected_proposal = await rejection_service.handle_user_request(
        "Research this prospect and give me risks, opportunities, and talking points."
    )
    assert rejected_proposal.approval_plan is not None
    rejected = build_user_action_response(
        await rejection_service.handle_user_action(
            _reject_event(
                plan_id=rejected_proposal.approval_plan.plan_id,
                surface_id=rejected_proposal.approval_plan.approval_surface_id or "",
                plan_version=rejected_proposal.approval_plan.plan_version,
            )
        )
    )
    error = build_error_response(ArtifactStorageError("artifact write failed"))

    # Assert
    payloads = [
        direct,
        clarification,
        plan_required,
        draft_updated,
        approved,
        rejected,
        error,
    ]
    assert [payload["status"] for payload in payloads] == [
        "direct",
        "clarification_required",
        "plan_required",
        "draft_updated",
        "approved",
        "rejected",
        "error",
    ]
    for payload in payloads:
        _assert_json_dict(payload)
    _assert_plan_contract(plan_required)
    _assert_plan_contract(draft_updated)
    assert approved["approvalResult"]["graphCreated"] is True
    assert approved["approvalResult"]["specialistsCalled"] is True
    assert rejected["approvalResult"]["graphCreated"] is False
    assert rejected["approvalResult"]["specialistsCalled"] is False


@pytest.mark.asyncio
async def test_adk_adapter_maps_approval_failures_to_safe_error_payloads() -> None:
    # Arrange
    adapter = AdkOrchestratorAdapter()
    tool_context = FakeToolContext()
    submitted = await adapter.submit_orchestrator_request(
        "Prepare me for tomorrow's meeting with ABC Manufacturing.",
        tool_context=tool_context,
    )

    # Act
    rejected = await adapter.approve_orchestrator_plan(
        submitted["planId"],
        submitted["approvalSurfaceId"],
        ["step_does_not_match_current_plan"],
        edited_plan_version=submitted["planVersion"],
        tool_context=tool_context,
    )

    # Assert
    _assert_json_dict(rejected)
    assert rejected["status"] == "error"
    assert rejected["error"]["code"] == "invalid_plan_mutation"
    rendered = json.dumps(rejected)
    assert "Traceback" not in rendered
    assert "step_does_not_match_current_plan" not in rendered


@pytest.mark.asyncio
async def test_adk_adapter_rejects_edited_plan_when_version_is_omitted() -> None:
    # Arrange
    adapter = AdkOrchestratorAdapter()
    tool_context = FakeToolContext()
    submitted = await adapter.submit_orchestrator_request(
        "Research this prospect and give me risks, opportunities, and talking points.",
        tool_context=tool_context,
    )
    updated = await adapter.add_plan_instruction(
        submitted["planId"],
        submitted["approvalSurfaceId"],
        submitted["stepIds"][0],
        "Focus the prep on near-term credit exposure.",
        edited_plan_version=submitted["planVersion"],
        tool_context=tool_context,
    )

    # Act
    rejected = await adapter.reject_orchestrator_plan(
        updated["planId"],
        updated["approvalSurfaceId"],
        "Too broad for this meeting.",
        tool_context=tool_context,
    )

    # Assert
    _assert_json_dict(rejected)
    assert updated["planVersion"] == submitted["planVersion"] + 1
    assert rejected["status"] == "rejected"
    assert rejected["planVersion"] == updated["planVersion"]
    assert rejected["approvalResult"]["graphCreated"] is False
    assert rejected["approvalResult"]["specialistsCalled"] is False
    assert rejected["approvalResult"]["reason"] == "Too broad for this meeting."


@pytest.mark.asyncio
async def test_adk_adapter_rejection_response_redacts_secret_like_reason() -> None:
    # Arrange
    adapter = AdkOrchestratorAdapter()
    tool_context = FakeToolContext()
    leaked_secret = "sk-or-v1-rejection-secret-should-not-leak"
    submitted = await adapter.submit_orchestrator_request(
        "Research this prospect and give me risks, opportunities, and talking points.",
        tool_context=tool_context,
    )

    # Act
    rejected = await adapter.reject_orchestrator_plan(
        submitted["planId"],
        submitted["approvalSurfaceId"],
        f"Too broad. api_key={leaked_secret}",
        edited_plan_version=submitted["planVersion"],
        tool_context=tool_context,
    )
    rendered = json.dumps(rejected, sort_keys=True)

    # Assert
    _assert_json_dict(rejected)
    assert rejected["status"] == "rejected"
    assert rejected["approvalResult"]["reason"] == "Too broad. <redacted-secret>"
    assert leaked_secret not in rendered
    assert "api_key" not in rendered


def test_error_builder_maps_known_exceptions_to_stable_safe_codes() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.approval_canvas import A2UIEmissionError
    from orchestrator_demo.a2ui_support.renderer_contract import RendererContractError
    from orchestrator_demo.orchestrator.approval_state import (
        PlanAlreadyFinalError,
        PlanMutationError,
        PlanNotFoundError,
        PlanSurfaceMismatchError,
        PlanVersionConflictError,
    )
    from orchestrator_demo.orchestrator.graph_runtime import (
        AdkGraphApiError,
        GraphRuntimeError,
    )
    from orchestrator_demo.orchestrator.surface_routes import SurfaceOwnershipError

    leaked_secret = "sk-or-v1-contract-secret-should-not-leak"
    cases = [
        (PlanNotFoundError(f"unknown plan {leaked_secret}"), "plan_not_found"),
        (PlanVersionConflictError("plan is version 2, got 1"), "stale_plan_version"),
        (PlanSurfaceMismatchError(f"wrong surface {leaked_secret}"), "surface_mismatch"),
        (PlanAlreadyFinalError("plan is already approved"), "plan_already_final"),
        (
            PlanMutationError("approvedStepIds must match current draft plan steps"),
            "invalid_plan_mutation",
        ),
        (GraphRuntimeError(f"handler failed {leaked_secret}"), "graph_execution_failed"),
        (
            AdkGraphApiError(
                "Traceback (most recent call last):\n"
                "File \"/tmp/adk_graph.py\", line 17, in build_workflow\n"
                f"RuntimeError: ADK graph API exposed {leaked_secret}"
            ),
            "graph_execution_failed",
        ),
        (
            SurfaceOwnershipError(f"surface registration failed {leaked_secret}"),
            "surface_ownership_error",
        ),
        (A2UIEmissionError(f"A2UI invalid {leaked_secret}"), "a2ui_delivery_error"),
        (
            RendererContractError(f"A2UI contract invalid {leaked_secret}"),
            "a2ui_delivery_error",
        ),
        (ArtifactStorageError(f"save failed {leaked_secret}"), "artifact_storage_error"),
    ]

    for exc, expected_code in cases:
        # Act
        payload = build_error_response(exc)

        # Assert
        _assert_json_dict(payload)
        assert payload["status"] == "error"
        assert payload["error"]["code"] == expected_code
        rendered = json.dumps(payload)
        assert leaked_secret not in rendered
        assert "Traceback" not in rendered
        assert 'File "/tmp/adk_graph.py"' not in rendered


def test_user_action_surface_route_error_redacts_route_payload() -> None:
    # Arrange
    leaked_secret = "sk-or-v1-surface-route-secret-should-not-leak"
    route_result = SurfaceRouteResult(
        status="error",
        surface_id="surface_contract_error",
        error={
            "code": "owner_handler_failed",
            "message": (
                "Traceback (most recent call last):\n"
                f"RuntimeError: handler exposed {leaked_secret}"
            ),
            "surfaceId": "surface_contract_error",
        },
        original_payload={
            "userAction": {
                "type": "select",
                "surfaceId": "surface_contract_error",
                "payload": {
                    "apiToken": leaked_secret,
                    "diagnostic": "File \"/tmp/app.py\", line 5, in handler",
                },
            }
        },
    )
    result = OrchestratorUserActionResult(
        status="error",
        surface_route_result=route_result,
    )

    # Act
    payload = build_user_action_response(result)

    # Assert
    _assert_json_dict(payload)
    assert payload["status"] == "error"
    rendered = json.dumps(payload)
    _assert_text_absent(rendered, leaked_secret, "secret-like diagnostic")
    _assert_text_absent(rendered, "Traceback", "traceback text")
    _assert_text_absent(rendered, 'File "/tmp/app.py"', "traceback frame text")
    assert payload["error"]["code"] == "surface_route_error"


def test_user_action_forwarded_surface_route_result_omits_raw_payload() -> None:
    # Arrange
    instruction_secret = "sk-or-v1-forwarded-instruction-should-not-leak"
    reason_secret = "sk-or-v1-forwarded-reason-should-not-leak"
    route_result = SurfaceRouteResult(
        status="forwarded",
        surface_id="specialist_surface_contract",
        owner=SurfaceOwner(
            surface_id="specialist_surface_contract",
            owner_type="specialist",
            owner_id="product_opportunity",
        ),
        response={"status": "handled"},
        original_payload={
            "userAction": {
                "type": "add_instruction",
                "surfaceId": "specialist_surface_contract",
                "payload": {
                    "instruction": f"Please include {instruction_secret}",
                    "reason": f"Rejected because {reason_secret}",
                },
            }
        },
    )
    result = OrchestratorUserActionResult(
        status="forwarded",
        surface_route_result=route_result,
    )

    # Act
    payload = build_user_action_response(result)

    # Assert
    _assert_json_dict(payload)
    assert payload["status"] == "forwarded"
    assert payload["surfaceRouteResult"]["status"] == "forwarded"
    assert payload["surfaceRouteResult"]["owner"]["ownerId"] == "product_opportunity"
    rendered = json.dumps(payload)
    _assert_text_absent(rendered, instruction_secret, "forwarded instruction")
    _assert_text_absent(rendered, reason_secret, "forwarded reason")
    assert "originalPayload" not in rendered
    assert "original_payload" not in rendered
