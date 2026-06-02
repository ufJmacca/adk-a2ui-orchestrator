from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from orchestrator_demo.a2a_support.transport import DataPart, TextPart
from orchestrator_demo.contracts import (
    AgentDescriptor,
    IntentSuggestion,
    LlmIntentAssessment,
    SpecialistResponse,
)
from orchestrator_demo.agents import build_default_specialists
from orchestrator_demo.orchestrator.service import OrchestratorService
from orchestrator_demo.registry.agent_registry import AgentRegistry


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


class InvalidA2uiSpecialist:
    agent_id = "product_opportunity"

    def __init__(self) -> None:
        self.call_count = 0
        self.calls = []

    async def handle(self, request: Any) -> SpecialistResponse:
        self.call_count += 1
        self.calls.append(request)
        return SpecialistResponse(
            response_id="response_product_opportunity_invalid_a2ui",
            agent_id=self.agent_id,
            content="Product Opportunity Agent: product fit summary.",
            structured_output={"summary": "product fit summary"},
            a2ui_payload={
                "version": "v0.9",
                "updateComponents": {
                    "surfaceId": "surface_product_recommendation",
                    "components": [],
                },
            },
            surface_id="surface_product_recommendation",
        )


class ExistingUserActionSpecialist:
    agent_id = "product_opportunity"

    def __init__(self) -> None:
        self.call_count = 0
        self.calls: list[Any] = []
        self.received_user_actions: list[Any] = []

    async def handle(self, request: Any) -> SpecialistResponse:
        self.call_count += 1
        self.calls.append(request)
        surface_id = "surface_existing_product_opportunity"
        return SpecialistResponse(
            response_id="response_product_opportunity_existing_handler",
            agent_id=self.agent_id,
            content="Product Opportunity Agent: product fit summary.",
            structured_output={"summary": "product fit summary"},
            a2ui_payload=[
                {
                    "version": "v0.9",
                    "createSurface": {
                        "surfaceId": surface_id,
                        "catalogId": (
                            "https://a2ui.org/specification/v0_9/basic_catalog.json"
                        ),
                    },
                },
                {
                    "version": "v0.9",
                    "updateComponents": {
                        "surfaceId": surface_id,
                        "components": [
                            {
                                "component": "Text",
                                "id": "root",
                                "text": "Treasury services fit the stated need.",
                            }
                        ],
                    },
                },
            ],
            surface_id=surface_id,
        )

    async def handle_user_action(self, user_action: Any) -> SpecialistResponse:
        self.received_user_actions.append(user_action)
        return SpecialistResponse(
            response_id="response_product_opportunity_existing_user_action",
            agent_id=self.agent_id,
            content="Product Opportunity Agent: existing user action handled.",
            structured_output={"status": "handled_by_existing_specialist"},
        )


class DeleteSurfaceUserActionSpecialist(ExistingUserActionSpecialist):
    async def handle_user_action(self, user_action: Any) -> SpecialistResponse:
        self.received_user_actions.append(user_action)
        return SpecialistResponse(
            response_id="response_product_opportunity_delete_surface",
            agent_id=self.agent_id,
            content="Product Opportunity Agent: surface closed.",
            structured_output={"status": "closed"},
            a2ui_payload=[
                {
                    "version": "v0.9",
                    "deleteSurface": {
                        "surfaceId": "surface_existing_product_opportunity",
                    },
                }
            ],
            surface_id="surface_existing_product_opportunity",
        )


def _descriptor_source(agent_id: str, *, display_name: str | None = None) -> str:
    display_name = display_name or agent_id.replace("_", " ").title()
    return f"""AgentDescriptor(
        agent_id={agent_id!r},
        display_name={display_name!r},
        capabilities=["business banking support"],
        input_schema={{"type": "object"}},
        output_schema={{"type": "object"}},
        a2ui_catalogs=["basic"],
        routing_examples=["Handle a {agent_id} request."],
        execution_mode="local_llm",
    )"""


def _write_registry_config(path: Path, agent_ids: Sequence[str]) -> None:
    path.write_text(
        "from orchestrator_demo.contracts import AgentDescriptor\n\n"
        "AVAILABLE_AGENTS = [\n"
        + ",\n".join(_descriptor_source(agent_id) for agent_id in agent_ids)
        + "\n]\n",
        encoding="utf-8",
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
async def test_direct_request_preserves_invalid_a2ui_text_fallback() -> None:
    # Arrange
    specialists = build_default_specialists()
    specialists["product_opportunity"] = InvalidA2uiSpecialist()
    slm_client = RecordingSlmIntentClient(
        IntentSuggestion(intent="product_opportunity", confidence=0.95)
    )
    intent_classifier = RecordingIntentClassifier(
        LlmIntentAssessment(
            intents=["product_opportunity"],
            confidence=0.95,
            complexity="simple",
            required_agents=["product_opportunity"],
            rationale="Injected single-agent assessment.",
        )
    )
    service = OrchestratorService(
        specialists=specialists,
        slm_client=slm_client,
        intent_classifier=intent_classifier,
    )

    # Act
    result = await service.handle_user_request("Suggest product opportunities.")

    # Assert
    assert result.path == "direct"
    assert result.specialist_responses[0].agent_id == "product_opportunity"
    assert result.final_artifacts["final_response"].agent_id == "product_opportunity"
    assert result.specialist_responses[0].a2ui_payload is None
    assert result.final_artifacts["final_response"].a2ui_payload is None
    assert len(result.a2ui_parts) == 1
    assert isinstance(result.a2ui_parts[0], TextPart)
    assert result.a2ui_parts[0].metadata["developerDiagnostic"]["fallback"] == "text"


@pytest.mark.asyncio
async def test_downstream_user_action_uses_existing_specialist_handler_by_default() -> None:
    # Arrange
    product_specialist = ExistingUserActionSpecialist()
    specialists = build_default_specialists()
    specialists["product_opportunity"] = product_specialist
    service = OrchestratorService(
        specialists=specialists,
        slm_client=RecordingSlmIntentClient(
            IntentSuggestion(intent="product_opportunity", confidence=0.95)
        ),
        intent_classifier=RecordingIntentClassifier(
            LlmIntentAssessment(
                intents=["product_opportunity"],
                confidence=0.95,
                complexity="simple",
                required_agents=["product_opportunity"],
                rationale="Injected single-agent assessment.",
            )
        ),
    )
    result = await service.handle_user_request("Suggest product opportunities.")
    surface_id = result.specialist_responses[0].surface_id
    assert surface_id is not None
    user_action = {
        "userAction": {
            "type": "specialist_action",
            "surfaceId": surface_id,
            "payload": {"action": "show_more_detail"},
        }
    }

    # Act
    routed = await service.handle_user_action(user_action)

    # Assert
    assert routed.status == "forwarded"
    assert product_specialist.received_user_actions == [user_action]
    assert [response.structured_output for response in routed.specialist_responses] == [
        {"status": "handled_by_existing_specialist"}
    ]


@pytest.mark.asyncio
async def test_downstream_delete_surface_user_action_is_returned_after_routing() -> None:
    # Arrange
    product_specialist = DeleteSurfaceUserActionSpecialist()
    specialists = build_default_specialists()
    specialists["product_opportunity"] = product_specialist
    service = OrchestratorService(
        specialists=specialists,
        slm_client=RecordingSlmIntentClient(
            IntentSuggestion(intent="product_opportunity", confidence=0.95)
        ),
        intent_classifier=RecordingIntentClassifier(
            LlmIntentAssessment(
                intents=["product_opportunity"],
                confidence=0.95,
                complexity="simple",
                required_agents=["product_opportunity"],
                rationale="Injected single-agent assessment.",
            )
        ),
    )
    requested = await service.handle_user_request("Suggest product opportunities.")
    surface_id = requested.specialist_responses[0].surface_id
    assert surface_id == "surface_existing_product_opportunity"
    assert service.surface_owner(surface_id) is not None
    user_action = {
        "userAction": {
            "type": "specialist_action",
            "surfaceId": surface_id,
            "payload": {"buttonId": "close"},
        }
    }

    # Act
    routed = await service.handle_user_action(user_action)

    # Assert
    assert routed.status == "forwarded"
    assert routed.specialist_responses[0].a2ui_payload is not None
    assert len(routed.a2ui_parts) == 1
    assert isinstance(routed.a2ui_parts[0], DataPart)
    assert routed.a2ui_parts[0].data["deleteSurface"]["surfaceId"] == surface_id
    assert service.surface_owner(surface_id) is None
    assert product_specialist.received_user_actions == [user_action]


@pytest.mark.asyncio
async def test_direct_request_missing_specialist_handler_returns_clarification() -> None:
    # Arrange
    specialists = build_default_specialists()
    specialists.pop("product_opportunity")
    slm_client = RecordingSlmIntentClient(
        IntentSuggestion(intent="product_opportunity", confidence=0.95)
    )
    intent_classifier = RecordingIntentClassifier(
        LlmIntentAssessment(
            intents=["product_opportunity"],
            confidence=0.96,
            complexity="simple",
            required_agents=["product_opportunity"],
            rationale="The registry advertises this agent, but no handler is wired.",
        )
    )
    service = OrchestratorService(
        specialists=specialists,
        slm_client=slm_client,
        intent_classifier=intent_classifier,
    )

    # Act
    result = await service.handle_user_request("Suggest product opportunities.")

    # Assert
    assert result.path == "clarification_required"
    assert result.decision.path == "clarification_required"
    assert result.decision.selected_agent is None
    assert "unavailable" in result.decision.reason.casefold()
    assert "product_opportunity" in result.decision.reason
    assert result.specialist_responses == ()
    assert result.approval_plan is None
    assert service.specialist_call_counts() == {}


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
        (
            LlmIntentAssessment(
                intents=["meeting_prep"],
                confidence=0.96,
                complexity="simple",
                required_agents=["synthesis"],
                rationale="Injected assessment has synthesis but no workstream.",
            ),
            "clarification_required",
            [],
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
    elif expected_path == "plan_required":
        assert result.specialist_responses == ()
        assert result.approval_plan is not None
        assert result.approval_plan.selected_agents == expected_agent_ids
    else:
        assert result.specialist_responses == ()
        assert result.approval_plan is None
        assert result.approval_result is None


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
async def test_failed_graph_approval_rolls_back_context_for_retry() -> None:
    # Arrange
    specialists = build_default_specialists()
    specialists.pop("relationship_summary")
    service = OrchestratorService(specialists=specialists)
    proposed = await service.handle_user_request(
        "Prepare me for tomorrow's meeting with ABC Manufacturing."
    )
    assert proposed.approval_plan is not None
    step_ids = [step.step_id for step in proposed.approval_plan.steps]
    first_step_id = proposed.approval_plan.steps[0].step_id

    # Act
    failed = await service.handle_user_action(
        _approve_event(
            proposed.approval_plan.plan_id,
            proposed.approval_plan.approval_surface_id or "",
            step_ids,
        )
    )

    # Assert
    assert failed.status == "failed"
    assert failed.approval_result is not None
    assert "no specialist handler registered" in (
        failed.approval_result.failure_reason or ""
    )
    assert [event.status for event in failed.status_events] == [
        "plan_approved",
        "graph_created",
        "step_failed",
    ]
    record_after_failed_approval = service.approval_record(
        proposed.approval_plan.plan_id
    )
    assert record_after_failed_approval.status == "draft"
    assert record_after_failed_approval.approved_plan is None

    edited = await service.handle_user_action(
        _add_instruction_event(
            proposed.approval_plan.plan_id,
            proposed.approval_plan.approval_surface_id or "",
            step_id=first_step_id,
        )
    )
    assert edited.status == "draft_updated"

    reapproval = await service.handle_user_action(
        _approve_event(
            proposed.approval_plan.plan_id,
            proposed.approval_plan.approval_surface_id or "",
            step_ids,
            plan_version=2,
        )
    )
    assert reapproval.status == "failed"
    assert reapproval.approval_result is not None
    assert "no specialist handler registered" in (
        reapproval.approval_result.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_approved_plan_rechecks_live_registry_before_graph_execution(
    tmp_path: Path,
) -> None:
    # Arrange
    config_path = tmp_path / "agent_config.py"
    _write_registry_config(
        config_path,
        [
            "relationship_summary",
            "internal_knowledge",
            "industry_research",
            "synthesis",
        ],
    )
    registry = AgentRegistry.from_config_path(config_path)
    service = OrchestratorService(registry=registry)
    proposed = await service.handle_user_request(
        "Prepare me for tomorrow's meeting with ABC Manufacturing."
    )
    assert proposed.approval_plan is not None
    _write_registry_config(
        config_path,
        [
            "relationship_summary",
            "internal_knowledge",
            "synthesis",
        ],
    )
    registry.reload()

    # Act
    result = await service.handle_user_action(
        _approve_event(
            proposed.approval_plan.plan_id,
            proposed.approval_plan.approval_surface_id or "",
            [step.step_id for step in proposed.approval_plan.steps],
        )
    )

    # Assert
    assert result.status == "failed"
    assert result.approval_result is not None
    assert result.approval_result.graph_created is True
    assert result.approval_result.specialists_called is False
    assert "industry_research" in (result.approval_result.failure_reason or "")
    assert [event.status for event in result.status_events] == [
        "plan_approved",
        "graph_created",
        "step_failed",
    ]
    failed_event = result.status_events[-1]
    assert failed_event.details["agentId"] == "industry_research"
    assert "Register agent industry_research" in failed_event.details[
        "developerMessage"
    ]
    assert service.specialist_call_counts() == {}
    record = service.approval_record(proposed.approval_plan.plan_id)
    assert record.status == "draft"
    assert record.approved_plan is None


@pytest.mark.asyncio
async def test_replace_agent_uses_registry_descriptors_added_after_service_construction(
    tmp_path: Path,
) -> None:
    # Arrange
    config_path = tmp_path / "agent_config.py"
    _write_registry_config(
        config_path,
        [
            "relationship_summary",
            "internal_knowledge",
            "industry_research",
            "synthesis",
        ],
    )
    registry = AgentRegistry.from_config_path(config_path)
    service = OrchestratorService(registry=registry)
    proposed = await service.handle_user_request(
        "Prepare me for tomorrow's meeting with ABC Manufacturing."
    )
    assert proposed.approval_plan is not None
    _write_registry_config(
        config_path,
        [
            "relationship_summary",
            "internal_knowledge",
            "industry_research",
            "credit_risk",
            "synthesis",
        ],
    )
    registry.reload()
    step_id = proposed.approval_plan.steps[0].step_id

    # Act
    edited = await service.handle_user_action(
        {
            "userAction": {
                "type": "replace_agent",
                "surfaceId": proposed.approval_plan.approval_surface_id or "",
                "payload": {
                    "planId": proposed.approval_plan.plan_id,
                    "editedPlanVersion": proposed.approval_plan.plan_version,
                    "stepId": step_id,
                    "replacementAgentId": "credit_risk",
                },
            }
        }
    )

    # Assert
    assert edited.status == "draft_updated"
    record = service.approval_record(proposed.approval_plan.plan_id)
    assert record.draft_plan.plan_version == 2
    assert record.draft_plan.steps[0].agent_id == "credit_risk"
    assert "credit_risk" in record.draft_plan.selected_agents


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
async def test_default_local_a2a_wrapper_handles_execution_and_user_actions() -> None:
    # Arrange
    from orchestrator_demo.a2a_support.local_remote_wrapper import LocalRemoteAgentWrapper

    service = OrchestratorService()
    assert isinstance(
        service._specialists["internal_knowledge"], LocalRemoteAgentWrapper
    )
    assert isinstance(
        service._specialists["product_opportunity"], LocalRemoteAgentWrapper
    )
    assert (
        service._specialist_user_action_adapters["product_opportunity"]
        is service._specialists["product_opportunity"]
    )

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
    assert routed.specialist_responses[0].structured_output["request_id"] == (
        "request_product_opportunity_user_action_1"
    )
    assert routed.final_artifacts["final_response"].agent_id == "product_opportunity"
    product_wrapper = service._specialists["product_opportunity"]
    assert isinstance(product_wrapper, LocalRemoteAgentWrapper)
    assert product_wrapper.calls[-1].context["user_action_payload"] == {
        "userAction": {
            "type": "specialist_action",
            "surfaceId": surface_id,
            "payload": {
                "agentId": "internal_knowledge",
                "action": "show_more_detail",
            },
        }
    }
