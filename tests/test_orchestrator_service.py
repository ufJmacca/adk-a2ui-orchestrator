from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Any

import pytest

from orchestrator_demo.a2a_support.transport import DataPart
from orchestrator_demo.contracts import (
    AgentDescriptor,
    IntentSuggestion,
    LlmIntentAssessment,
    SpecialistResponse,
)
from orchestrator_demo.orchestrator.service import OrchestratorService


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


class RecordingUserActionAdapter:
    def __init__(self) -> None:
        self.received_user_actions: list[Any] = []

    async def handle_user_action(self, user_action: Any) -> SpecialistResponse:
        self.received_user_actions.append(user_action)
        return SpecialistResponse(
            response_id="response_product_opportunity_user_action",
            agent_id="product_opportunity",
            content="Product Opportunity Agent: user action handled.",
            structured_output={"status": "handled"},
        )


def _approve_event(plan_id: str, surface_id: str, step_ids: list[str]) -> dict[str, Any]:
    return {
        "userAction": {
            "type": "approve_plan",
            "surfaceId": surface_id,
            "payload": {
                "planId": plan_id,
                "editedPlanVersion": 1,
                "approvedStepIds": step_ids,
            },
        }
    }


def _reject_event(
    plan_id: str,
    surface_id: str,
    *,
    plan_version: int = 1,
) -> dict[str, Any]:
    return {
        "userAction": {
            "type": "reject_plan",
            "surfaceId": surface_id,
            "payload": {
                "planId": plan_id,
                "editedPlanVersion": plan_version,
                "reason": "Too broad; focus on credit risk only.",
            },
        }
    }


def _add_instruction_event(
    plan_id: str,
    surface_id: str,
    *,
    step_id: str,
) -> dict[str, Any]:
    return {
        "userAction": {
            "type": "add_instruction",
            "surfaceId": surface_id,
            "payload": {
                "planId": plan_id,
                "editedPlanVersion": 1,
                "stepId": step_id,
                "instruction": "Prioritize covenant follow-ups.",
            },
        }
    }


def _action_contexts_by_type(
    a2ui_part: DataPart,
) -> dict[str, dict[str, Any]]:
    components = a2ui_part.data["updateComponents"]["components"]
    contexts: dict[str, dict[str, Any]] = {}
    for component in components:
        event = component.get("action", {}).get("event")
        if not isinstance(event, dict):
            continue
        context = event.get("context")
        if isinstance(context, dict) and isinstance(context.get("type"), str):
            contexts[context["type"]] = context
    return contexts


@pytest.mark.asyncio
async def test_simple_direct_request_returns_one_specialist_response_no_approval_ui() -> None:
    # Arrange
    service = OrchestratorService()
    user_input = "Summarize the internal notes for ABC Manufacturing."

    # Act
    result = await service.handle_user_request(user_input)

    # Assert
    assert result.path == "direct"
    assert result.approval_plan is None
    assert result.approval_result is None
    assert result.graph_execution is None
    assert result.a2ui_parts == ()
    assert [response.agent_id for response in result.specialist_responses] == [
        "internal_knowledge"
    ]
    assert service.specialist_call_counts() == {"internal_knowledge": 1}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("assessment", "expected_path", "expected_agent_ids"),
    [
        (
            LlmIntentAssessment(
                intents=["internal_knowledge"],
                confidence=0.95,
                complexity="simple",
                required_agents=["internal_knowledge"],
                rationale="Injected single-agent assessment.",
            ),
            "direct",
            ["internal_knowledge"],
        ),
        (
            LlmIntentAssessment(
                intents=["meeting_prep", "relationship_summary"],
                confidence=0.94,
                complexity="complex",
                required_agents=[
                    "relationship_summary",
                    "internal_knowledge",
                    "synthesis",
                ],
                rationale="Injected multi-agent assessment.",
            ),
            "plan_required",
            [
                "relationship_summary",
                "internal_knowledge",
                "synthesis",
            ],
        ),
    ],
)
async def test_handle_user_request_uses_injected_classification_once_to_choose_path(
    assessment: LlmIntentAssessment,
    expected_path: str,
    expected_agent_ids: list[str],
) -> None:
    # Arrange
    user_input = "Use the injected classifier result for this request."
    slm_client = RecordingSlmIntentClient(
        IntentSuggestion(intent="meeting_prep", confidence=0.9)
    )
    intent_classifier = RecordingIntentClassifier(assessment)
    service = OrchestratorService(
        slm_client=slm_client,
        intent_classifier=intent_classifier,
    )

    # Act
    result = await service.handle_user_request(user_input)

    # Assert
    assert slm_client.inputs == [user_input]
    assert len(intent_classifier.calls) == 1
    classifier_call = intent_classifier.calls[0]
    assert classifier_call["user_input"] == user_input
    assert classifier_call["slm_suggestion"] == slm_client.suggestion
    assert set(classifier_call["available_agent_ids"]) == {
        "industry_research",
        "web_search",
        "internal_knowledge",
        "credit_risk",
        "relationship_summary",
        "product_opportunity",
        "compliance_policy",
        "data_quality",
        "meeting_prep",
        "synthesis",
    }
    assert result.path == expected_path
    if expected_path == "direct":
        assert [response.agent_id for response in result.specialist_responses] == (
            expected_agent_ids
        )
        assert result.approval_plan is None
    else:
        assert result.specialist_responses == ()
        assert result.approval_plan is not None
        assert result.approval_plan.selected_agents == expected_agent_ids


@pytest.mark.asyncio
async def test_complex_request_returns_approval_plan_before_specialist_call() -> None:
    # Arrange
    service = OrchestratorService()
    user_input = "Prepare me for tomorrow's meeting with ABC Manufacturing."

    # Act
    result = await service.handle_user_request(user_input)

    # Assert
    assert result.path == "plan_required"
    assert result.approval_plan is not None
    assert result.approval_plan.selected_agents == [
        "relationship_summary",
        "internal_knowledge",
        "industry_research",
        "synthesis",
    ]
    assert result.specialist_responses == ()
    assert service.specialist_call_counts() == {}
    assert len(result.a2ui_parts) == 2
    assert all(isinstance(part, DataPart) for part in result.a2ui_parts)
    assert all(part.mime_type == "application/json+a2ui" for part in result.a2ui_parts)
    create_surface, update_components = result.a2ui_parts
    approval_surface_id = result.approval_plan.approval_surface_id
    assert create_surface.data["createSurface"]["surfaceId"] == approval_surface_id
    assert update_components.data["updateComponents"]["surfaceId"] == (
        approval_surface_id
    )
    contexts_by_type = _action_contexts_by_type(update_components)
    assert set(contexts_by_type) >= {"approve_plan", "reject_plan", "edit_plan"}
    for action_type in ("approve_plan", "reject_plan", "edit_plan"):
        context = contexts_by_type[action_type]
        payload = context["payload"]
        assert context["surfaceId"] == approval_surface_id
        assert payload["planId"] == result.approval_plan.plan_id
        assert payload["planVersion"] == result.approval_plan.plan_version
        assert payload["editedPlanVersion"] == result.approval_plan.plan_version
    assert contexts_by_type["approve_plan"]["payload"]["approvedStepIds"] == [
        step.step_id for step in result.approval_plan.steps
    ]
    assert contexts_by_type["reject_plan"]["payload"]["reason"] == ""
    assert contexts_by_type["edit_plan"]["payload"]["editableFields"] == [
        "steps",
        "selectedAgents",
    ]
    owner = service.surface_owner(result.approval_plan.approval_surface_id or "")
    assert owner is not None
    assert owner.owner_type == "orchestrator"
    assert owner.plan_id == result.approval_plan.plan_id


@pytest.mark.asyncio
async def test_approval_action_freezes_plan_executes_graph_and_returns_artifacts() -> None:
    # Arrange
    service = OrchestratorService()
    proposed = await service.handle_user_request(
        "Prepare me for tomorrow's meeting with ABC Manufacturing."
    )
    assert proposed.approval_plan is not None
    event = _approve_event(
        proposed.approval_plan.plan_id,
        proposed.approval_plan.approval_surface_id or "",
        [step.step_id for step in proposed.approval_plan.steps],
    )

    # Act
    result = await service.handle_user_action(event)

    # Assert
    assert result.status == "approved"
    assert result.approval_result is not None
    assert result.approval_result.approved_plan == proposed.approval_plan
    assert result.graph_execution is not None
    assert [event.status for event in result.status_events] == [
        "plan_approved",
        "graph_created",
        "parallel_branch_started",
        "step_started",
        "step_completed",
        "parallel_branch_completed",
        "parallel_branch_started",
        "step_started",
        "step_completed",
        "parallel_branch_completed",
        "parallel_branch_started",
        "step_started",
        "step_completed",
        "parallel_branch_completed",
        "synthesis_started",
        "step_started",
        "step_completed",
        "final_response_ready",
    ]
    assert [response.agent_id for response in result.specialist_responses] == [
        "relationship_summary",
        "internal_knowledge",
        "industry_research",
        "synthesis",
    ]
    assert service.specialist_call_counts() == {
        "relationship_summary": 1,
        "internal_knowledge": 1,
        "industry_research": 1,
        "synthesis": 1,
    }
    record = service.approval_record(proposed.approval_plan.plan_id)
    assert record.status == "approved"
    assert record.approved_plan == proposed.approval_plan
    assert result.final_artifacts["final_response"].agent_id == "synthesis"


@pytest.mark.asyncio
async def test_edit_and_reject_actions_follow_approval_state_rules() -> None:
    # Arrange
    service = OrchestratorService()
    proposed = await service.handle_user_request(
        "Prepare me for tomorrow's meeting with ABC Manufacturing."
    )
    assert proposed.approval_plan is not None
    first_step_id = proposed.approval_plan.steps[0].step_id

    # Act
    edited = await service.handle_user_action(
        _add_instruction_event(
            proposed.approval_plan.plan_id,
            proposed.approval_plan.approval_surface_id or "",
            step_id=first_step_id,
        )
    )
    rejected = await service.handle_user_action(
        _reject_event(
            proposed.approval_plan.plan_id,
            proposed.approval_plan.approval_surface_id or "",
            plan_version=2,
        )
    )

    # Assert
    assert edited.status == "draft_updated"
    assert edited.graph_execution is None
    assert edited.specialist_responses == ()
    assert len(edited.a2ui_parts) == 2
    assert service.specialist_call_counts() == {}

    assert rejected.status == "rejected"
    assert rejected.graph_execution is None
    assert rejected.specialist_responses == ()
    assert service.specialist_call_counts() == {}
    record = service.approval_record(proposed.approval_plan.plan_id)
    assert record.status == "rejected"
    assert record.rejection_reason == "Too broad; focus on credit risk only."


@pytest.mark.asyncio
async def test_downstream_specialist_user_action_routes_by_surface_id_only() -> None:
    # Arrange
    adapter = RecordingUserActionAdapter()
    service = OrchestratorService(
        specialist_user_action_adapters={"product_opportunity": adapter}
    )
    result = await service.handle_user_request(
        "What product opportunities should I consider for a cafe business?"
    )
    assert result.specialist_responses[0].surface_id is not None
    surface_id = result.specialist_responses[0].surface_id
    user_action = {
        "userAction": {
            "type": "specialist_action",
            "surfaceId": surface_id,
            "payload": {
                "agentId": "internal_knowledge",
                "action": "show_more_detail",
            },
        }
    }
    original_user_action = deepcopy(user_action)

    # Act
    routed = await service.handle_user_action(user_action)

    # Assert
    assert routed.status == "forwarded"
    assert routed.surface_route_result is not None
    assert routed.surface_route_result.owner is not None
    assert routed.surface_route_result.owner.owner_id == "product_opportunity"
    assert adapter.received_user_actions == [user_action]
    assert adapter.received_user_actions[0] is user_action
    assert user_action == original_user_action


@pytest.mark.asyncio
async def test_default_specialist_user_action_adapter_returns_owner_response() -> None:
    # Arrange
    service = OrchestratorService()
    result = await service.handle_user_request(
        "What product opportunities should I consider for a cafe business?"
    )
    surface_id = result.specialist_responses[0].surface_id
    assert surface_id is not None

    # Act
    routed = await service.handle_user_action(
        {
            "userAction": {
                "type": "specialist_action",
                "surfaceId": surface_id,
                "payload": {
                    "agentId": "internal_knowledge",
                    "action": "show_more_detail",
                },
            }
        }
    )

    # Assert
    assert routed.status == "forwarded"
    assert routed.surface_route_result is not None
    assert routed.surface_route_result.owner is not None
    assert routed.surface_route_result.owner.owner_id == "product_opportunity"
    assert [response.agent_id for response in routed.specialist_responses] == [
        "product_opportunity"
    ]
    assert routed.final_artifacts["final_response"].agent_id == "product_opportunity"
