from __future__ import annotations

import json
from collections.abc import Sequence
from copy import deepcopy
from typing import Any

import pytest

from orchestrator_demo.a2ui_support.secret_safety import REDACTED_SECRET
from orchestrator_demo.contracts import (
    AgentDescriptor,
    IntentSuggestion,
    LlmIntentAssessment,
)
from orchestrator_demo.orchestrator.service import OrchestratorService
from orchestrator_demo.orchestrator.session_snapshot import SNAPSHOT_SCHEMA_VERSION


KNOWN_SECRET = "OPENROUTER_API_KEY=sk-or-v1-service-hydration-secret"


class RecordingSlmIntentClient:
    def __init__(self, suggestion: IntentSuggestion) -> None:
        self.suggestion = suggestion
        self.inputs: list[str] = []

    async def classify(self, user_input: str) -> IntentSuggestion:
        self.inputs.append(user_input)
        return self.suggestion


class RecordingIntentClassifier:
    def __init__(self, assessment: LlmIntentAssessment) -> None:
        self.assessment = assessment
        self.calls: list[dict[str, Any]] = []

    async def assess(
        self,
        user_input: str,
        slm_suggestion: IntentSuggestion,
        *,
        available_agents: Sequence[AgentDescriptor] | None = None,
    ) -> LlmIntentAssessment:
        self.calls.append(
            {
                "user_input": user_input,
                "slm_suggestion": slm_suggestion,
                "available_agent_ids": [
                    agent.agent_id for agent in available_agents or ()
                ],
            }
        )
        return self.assessment


def _complex_internal_knowledge_classifier() -> tuple[
    RecordingSlmIntentClient,
    RecordingIntentClassifier,
]:
    return (
        RecordingSlmIntentClient(
            IntentSuggestion(intent="meeting_prep", confidence=0.9)
        ),
        RecordingIntentClassifier(
            LlmIntentAssessment(
                intents=["meeting_prep"],
                confidence=0.94,
                complexity="complex",
                required_agents=["internal_knowledge", "synthesis"],
                rationale="Injected two-step workflow.",
            )
        ),
    )


def _approve_event(
    plan_id: str,
    surface_id: str,
    step_ids: list[str],
    *,
    plan_version: int = 1,
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


@pytest.mark.asyncio
async def test_complex_request_can_be_approved_after_service_snapshot_restore() -> None:
    # Arrange
    slm_client, intent_classifier = _complex_internal_knowledge_classifier()
    initial_service = OrchestratorService(
        slm_client=slm_client,
        intent_classifier=intent_classifier,
    )
    proposed = await initial_service.handle_user_request(
        "Prepare a focused internal-knowledge workflow."
    )
    assert proposed.approval_plan is not None
    plan = proposed.approval_plan
    approval_surface_id = plan.approval_surface_id or ""
    step_ids = [step.step_id for step in plan.steps]

    # Act
    snapshot = initial_service.export_session_snapshot()
    restored_service = OrchestratorService()
    restored_service.restore_session_snapshot(snapshot)
    approved = await restored_service.handle_user_action(
        _approve_event(plan.plan_id, approval_surface_id, step_ids)
    )

    # Assert
    assert snapshot["schemaVersion"] == SNAPSHOT_SCHEMA_VERSION
    assert set(snapshot) == {
        "schemaVersion",
        "approvalRecords",
        "requestContextsByPlanId",
        "surfaceRegistry",
        "artifactRefs",
    }
    assert snapshot["approvalRecords"][plan.plan_id]["status"] == "draft"
    assert snapshot["requestContextsByPlanId"][plan.plan_id]["draftPlanId"] == (
        plan.plan_id
    )
    assert snapshot["surfaceRegistry"]["ownersBySurfaceId"][approval_surface_id][
        "planId"
    ] == plan.plan_id
    assert approved.status == "approved"
    assert approved.graph_execution is not None
    assert [response.agent_id for response in approved.specialist_responses] == [
        "internal_knowledge",
        "synthesis",
    ]
    assert initial_service.specialist_call_counts() == {}
    assert restored_service.specialist_call_counts() == {
        "internal_knowledge": 1,
        "synthesis": 1,
    }
    restored_record = restored_service.approval_record(plan.plan_id)
    assert restored_record.status == "approved"
    assert restored_record.approved_plan == plan


@pytest.mark.asyncio
async def test_restore_round_trips_artifact_refs_without_leaking_secrets() -> None:
    # Arrange
    slm_client, intent_classifier = _complex_internal_knowledge_classifier()
    initial_service = OrchestratorService(
        slm_client=slm_client,
        intent_classifier=intent_classifier,
    )
    proposed = await initial_service.handle_user_request(
        "Prepare a focused internal-knowledge workflow."
    )
    assert proposed.approval_plan is not None
    snapshot = initial_service.export_session_snapshot()
    snapshot_with_artifacts = deepcopy(snapshot)
    snapshot_with_artifacts["artifactRefs"] = {
        "orchestrator_latest_result.json": {
            "uri": "file://.adk/artifacts/orchestrator_latest_result.json",
            "diagnostic": KNOWN_SECRET,
        }
    }

    # Act
    restored_service = OrchestratorService()
    restored_service.restore_session_snapshot(snapshot_with_artifacts)
    restored_snapshot = restored_service.export_session_snapshot()
    serialized = json.dumps(restored_snapshot, sort_keys=True)

    # Assert
    assert restored_snapshot["artifactRefs"]["orchestrator_latest_result.json"][
        "uri"
    ] == "file://.adk/artifacts/orchestrator_latest_result.json"
    assert KNOWN_SECRET not in serialized
    assert REDACTED_SECRET in serialized
