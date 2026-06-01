from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from orchestrator_demo.a2a_support.transport import DataPart
from orchestrator_demo.a2ui_support.schema_manager import A2UI_VERSION, BASIC_CATALOG_ID
from orchestrator_demo.contracts import (
    AgentDescriptor,
    ExecutionPlan,
    IntentSuggestion,
    LlmIntentAssessment,
    PlanStep,
    SpecialistRequest,
    SpecialistResponse,
)
from orchestrator_demo.orchestrator.approval_state import PlanMutationError
from orchestrator_demo.orchestrator.graph_runtime import (
    GraphExecutionResult,
    OwnedSpecialistResponse,
    build_graph_spec,
)
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


class MisreportingA2uiUserActionAdapter:
    async def handle_user_action(self, _user_action: Any) -> SpecialistResponse:
        surface_id = "surface_product_opportunity_detail"
        return SpecialistResponse(
            response_id="response_product_opportunity_detail",
            agent_id="internal_knowledge",
            content="Product Opportunity Agent: follow-up detail.",
            structured_output={"status": "handled"},
            surface_id=surface_id,
            a2ui_payload=[
                {
                    "version": A2UI_VERSION,
                    "createSurface": {
                        "surfaceId": surface_id,
                        "catalogId": BASIC_CATALOG_ID,
                    }
                },
                {
                    "version": A2UI_VERSION,
                    "updateComponents": {
                        "surfaceId": surface_id,
                        "components": [
                            {
                                "id": "detail",
                                "component": "Text",
                                "text": "Follow-up detail.",
                            }
                        ],
                    }
                },
            ],
        )


class RecordingDynamicSpecialist:
    def __init__(self, agent_id: str) -> None:
        self._agent_id = agent_id
        self.call_count = 0
        self.calls: list[SpecialistRequest] = []

    @property
    def agent_id(self) -> str:
        return self._agent_id

    async def handle(self, request: SpecialistRequest) -> SpecialistResponse:
        self.call_count += 1
        self.calls.append(request)
        return SpecialistResponse(
            response_id=f"response_{request.request_id.removeprefix('request_')}",
            agent_id=request.agent_id,
            content="Dynamic specialist handled.",
        )


class RecordingA2uiActionSpecialist(RecordingDynamicSpecialist):
    def __init__(self, agent_id: str) -> None:
        super().__init__(agent_id)
        self.received_user_actions: list[Any] = []

    async def handle(self, request: SpecialistRequest) -> SpecialistResponse:
        self.call_count += 1
        self.calls.append(request)
        response_suffix = request.request_id.removeprefix("request_")
        surface_id = f"surface_{response_suffix}"
        return SpecialistResponse(
            response_id=f"response_{response_suffix}",
            agent_id=request.agent_id,
            content="Dynamic specialist returned A2UI.",
            a2ui_payload=_specialist_a2ui_payload(surface_id),
            surface_id=surface_id,
        )

    async def handle_user_action(self, user_action: Any) -> SpecialistResponse:
        self.received_user_actions.append(user_action)
        return SpecialistResponse(
            response_id=f"response_{self.agent_id}_user_action_forwarded",
            agent_id=self.agent_id,
            content="Dynamic specialist handled forwarded A2UI user action.",
            structured_output={"status": "forwarded_by_injected_specialist"},
        )


class StaticRegistry:
    def __init__(self, descriptors: Sequence[AgentDescriptor]) -> None:
        self._descriptors = [descriptor.model_copy(deep=True) for descriptor in descriptors]

    def descriptors(self) -> list[AgentDescriptor]:
        return [
            descriptor.model_copy(deep=True) for descriptor in self._descriptors
        ]


def _descriptor(agent_id: str) -> AgentDescriptor:
    return AgentDescriptor(
        agent_id=agent_id,
        display_name=agent_id.replace("_", " ").title(),
        capabilities=["business banking support"],
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        a2ui_catalogs=["basic"],
        routing_examples=[f"Handle a {agent_id} request."],
        execution_mode="local_llm",
    )


def _write_registry_config(path: Path, agent_ids: Sequence[str]) -> None:
    descriptor_sources = [
        repr(_descriptor(agent_id).model_dump(mode="json"))
        for agent_id in agent_ids
    ]
    path.write_text(
        "AVAILABLE_AGENTS = [\n"
        + ",\n".join(descriptor_sources)
        + "\n]\n",
        encoding="utf-8",
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


def _specialist_a2ui_payload(surface_id: str) -> list[dict[str, Any]]:
    return [
        {
            "version": A2UI_VERSION,
            "createSurface": {
                "surfaceId": surface_id,
                "catalogId": BASIC_CATALOG_ID,
            },
        },
        {
            "version": A2UI_VERSION,
            "updateComponents": {
                "surfaceId": surface_id,
                "components": [
                    {
                        "id": f"component_{surface_id}",
                        "component": "Text",
                        "text": "Specialist UI.",
                    }
                ],
            },
        },
    ]


def _replace_agent_event(
    plan_id: str,
    surface_id: str,
    *,
    step_id: str,
    replacement_agent_id: str,
    plan_version: int = 1,
) -> dict[str, Any]:
    return {
        "userAction": {
            "type": "replace_agent",
            "surfaceId": surface_id,
            "payload": {
                "planId": plan_id,
                "editedPlanVersion": plan_version,
                "stepId": step_id,
                "replacementAgentId": replacement_agent_id,
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


def test_graph_a2ui_surfaces_use_carried_owner_for_out_of_order_responses() -> None:
    # Arrange
    service = OrchestratorService()
    plan = ExecutionPlan(
        plan_id="plan_parallel_owner",
        objective="Prepare parallel owned specialist UIs.",
        detected_intents=["meeting_prep"],
        selected_agents=["relationship_summary", "product_opportunity"],
        steps=[
            PlanStep(
                step_id="step_slow_branch",
                agent_id="relationship_summary",
                instruction="Summarize the relationship.",
                expected_output="Relationship summary.",
                parallel_group="parallel_owner_check",
            ),
            PlanStep(
                step_id="step_fast_branch",
                agent_id="product_opportunity",
                instruction="Find product opportunities.",
                expected_output="Product opportunities.",
                parallel_group="parallel_owner_check",
            ),
        ],
        approval_surface_id="surface_plan_parallel_owner",
    )
    slow_request = SpecialistRequest(
        request_id="request_parallel_owner_slow",
        user_input="Summarize the relationship.",
        agent_id="relationship_summary",
        plan_id=plan.plan_id,
        step_id="step_slow_branch",
    )
    fast_request = SpecialistRequest(
        request_id="request_parallel_owner_fast",
        user_input="Find product opportunities.",
        agent_id="product_opportunity",
        plan_id=plan.plan_id,
        step_id="step_fast_branch",
    )
    fast_response = SpecialistResponse(
        response_id="response_parallel_owner_fast",
        agent_id="internal_knowledge",
        content="Product Opportunity Agent: fast branch completed.",
        surface_id="surface_product_opportunity_fast",
        a2ui_payload=_specialist_a2ui_payload("surface_product_opportunity_fast"),
    )
    slow_response = SpecialistResponse(
        response_id="response_parallel_owner_slow",
        agent_id="relationship_summary",
        content="Relationship Summary Agent: slow branch completed.",
        surface_id="surface_relationship_summary_slow",
        a2ui_payload=_specialist_a2ui_payload("surface_relationship_summary_slow"),
    )
    graph_execution = GraphExecutionResult(
        graph=build_graph_spec(plan),
        workflow=object(),
        status_events=(),
        specialist_requests=(slow_request, fast_request),
        specialist_responses=(fast_response, slow_response),
        owned_specialist_responses=(
            OwnedSpecialistResponse(
                owner_agent_id="product_opportunity",
                response=fast_response,
            ),
            OwnedSpecialistResponse(
                owner_agent_id="relationship_summary",
                response=slow_response,
            ),
        ),
        adk_event_outputs=(),
    )

    # Act
    parts = service._prepare_graph_response_a2ui(graph_execution)

    # Assert
    assert len(parts) == 4
    fast_owner = service.surface_owner("surface_product_opportunity_fast")
    assert fast_owner is not None
    assert fast_owner.owner_id == "product_opportunity"
    slow_owner = service.surface_owner("surface_relationship_summary_slow")
    assert slow_owner is not None
    assert slow_owner.owner_id == "relationship_summary"


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
async def test_injected_specialist_user_action_handler_is_used_by_default() -> None:
    # Arrange
    agent_id = "product_opportunity"
    specialist = RecordingA2uiActionSpecialist(agent_id)
    service = OrchestratorService(
        registry=StaticRegistry([_descriptor(agent_id)]),
        specialists={agent_id: specialist},
        slm_client=RecordingSlmIntentClient(
            IntentSuggestion(intent=agent_id, confidence=0.95)
        ),
        intent_classifier=RecordingIntentClassifier(
            LlmIntentAssessment(
                intents=[agent_id],
                confidence=0.95,
                complexity="simple",
                required_agents=[agent_id],
                rationale="Injected A2UI specialist route.",
            )
        ),
    )
    result = await service.handle_user_request(
        "What product opportunities should I consider for a cafe business?"
    )
    surface_id = result.specialist_responses[0].surface_id
    assert surface_id is not None
    user_action = {
        "userAction": {
            "type": "specialist_action",
            "surfaceId": surface_id,
            "payload": {"action": "show_more_detail"},
        }
    }
    original_user_action = deepcopy(user_action)

    # Act
    routed = await service.handle_user_action(user_action)

    # Assert
    assert routed.status == "forwarded"
    assert specialist.received_user_actions == [user_action]
    assert specialist.received_user_actions[0] is user_action
    assert user_action == original_user_action
    assert routed.specialist_responses[0].structured_output == {
        "status": "forwarded_by_injected_specialist"
    }
    assert specialist.call_count == 1


@pytest.mark.asyncio
async def test_repeated_direct_a2ui_specialist_requests_use_distinct_surfaces() -> None:
    # Arrange
    service = OrchestratorService()
    user_input = "What product opportunities should I consider for a cafe business?"

    # Act
    first = await service.handle_user_request(user_input)
    second = await service.handle_user_request(user_input)

    # Assert
    assert first.path == "direct"
    assert second.path == "direct"
    first_response = first.specialist_responses[0]
    second_response = second.specialist_responses[0]
    assert first_response.response_id != second_response.response_id
    assert first_response.surface_id is not None
    assert second_response.surface_id is not None
    assert first_response.surface_id != second_response.surface_id
    assert first.a2ui_parts[0].data["createSurface"]["surfaceId"] == (
        first_response.surface_id
    )
    assert second.a2ui_parts[0].data["createSurface"]["surfaceId"] == (
        second_response.surface_id
    )
    assert service.surface_owner(first_response.surface_id).owner_id == (
        "product_opportunity"
    )
    assert service.surface_owner(second_response.surface_id).owner_id == (
        "product_opportunity"
    )


@pytest.mark.asyncio
async def test_direct_request_sanitizes_dynamic_agent_id_in_generated_request_id() -> None:
    # Arrange
    dynamic_agent_id = "agent.v1"
    specialist = RecordingDynamicSpecialist(dynamic_agent_id)
    service = OrchestratorService(
        registry=StaticRegistry([_descriptor(dynamic_agent_id)]),
        specialists={dynamic_agent_id: specialist},
        slm_client=RecordingSlmIntentClient(
            IntentSuggestion(intent="internal_knowledge", confidence=0.95)
        ),
        intent_classifier=RecordingIntentClassifier(
            LlmIntentAssessment(
                intents=["internal_knowledge"],
                confidence=0.95,
                complexity="simple",
                required_agents=[dynamic_agent_id],
                rationale="Dynamic single-agent route.",
            )
        ),
    )

    # Act
    result = await service.handle_user_request("Handle the dynamic agent request.")

    # Assert
    assert result.path == "direct"
    assert [response.agent_id for response in result.specialist_responses] == [
        dynamic_agent_id
    ]
    assert specialist.call_count == 1
    assert specialist.calls[0].request_id.startswith("request_direct_agent_v1_")
    assert "." not in specialist.calls[0].request_id


@pytest.mark.asyncio
async def test_direct_route_without_specialist_handler_returns_structured_error() -> None:
    # Arrange
    selected_agent = "remote_agent"
    service = OrchestratorService(
        registry=StaticRegistry([_descriptor(selected_agent)]),
        slm_client=RecordingSlmIntentClient(
            IntentSuggestion(intent="internal_knowledge", confidence=0.95)
        ),
        intent_classifier=RecordingIntentClassifier(
            LlmIntentAssessment(
                intents=["internal_knowledge"],
                confidence=0.95,
                complexity="simple",
                required_agents=[selected_agent],
                rationale="Descriptor-only direct route.",
            )
        ),
    )

    # Act
    result = await service.handle_user_request("Handle this remote-only request.")

    # Assert
    assert result.path == "clarification_required"
    assert result.decision.path == "clarification_required"
    assert result.context.decision.path == "clarification_required"
    assert result.specialist_responses == ()
    assert result.a2ui_parts == ()
    assert result.final_artifacts["error"] == {
        "code": "specialist_handler_unavailable",
        "agent_id": selected_agent,
        "message": result.decision.reason,
    }
    assert selected_agent in result.decision.reason
    assert service.specialist_call_counts() == {}


@pytest.mark.asyncio
async def test_empty_specialist_mapping_does_not_install_default_handlers() -> None:
    # Arrange
    selected_agent = "internal_knowledge"
    service = OrchestratorService(
        registry=StaticRegistry([_descriptor(selected_agent)]),
        specialists={},
        slm_client=RecordingSlmIntentClient(
            IntentSuggestion(intent=selected_agent, confidence=0.95)
        ),
        intent_classifier=RecordingIntentClassifier(
            LlmIntentAssessment(
                intents=[selected_agent],
                confidence=0.95,
                complexity="simple",
                required_agents=[selected_agent],
                rationale="Intentional no-handler direct route.",
            )
        ),
    )

    # Act
    result = await service.handle_user_request("Summarize the internal notes.")

    # Assert
    assert result.path == "clarification_required"
    assert result.decision.path == "clarification_required"
    assert result.specialist_responses == ()
    assert result.a2ui_parts == ()
    assert result.final_artifacts["error"] == {
        "code": "specialist_handler_unavailable",
        "agent_id": selected_agent,
        "message": result.decision.reason,
    }
    assert service.specialist_call_counts() == {}


@pytest.mark.asyncio
async def test_complex_plan_without_specialist_handler_returns_structured_error() -> None:
    # Arrange
    selected_agent = "remote_agent"
    service = OrchestratorService(
        registry=StaticRegistry([_descriptor(selected_agent), _descriptor("synthesis")]),
        slm_client=RecordingSlmIntentClient(
            IntentSuggestion(intent="unknown", confidence=0.5)
        ),
        intent_classifier=RecordingIntentClassifier(
            LlmIntentAssessment(
                intents=["unknown"],
                confidence=0.9,
                complexity="complex",
                required_agents=[selected_agent],
                rationale="Descriptor-only complex route.",
            )
        ),
    )

    # Act
    result = await service.handle_user_request("Coordinate this remote-only review.")

    # Assert
    assert result.path == "clarification_required"
    assert result.decision.path == "clarification_required"
    assert result.context.decision.path == "clarification_required"
    assert result.context.draft_plan_id is None
    assert result.approval_plan is None
    assert result.specialist_responses == ()
    assert result.a2ui_parts == ()
    assert result.final_artifacts["error"] == {
        "code": "specialist_handler_unavailable",
        "agent_ids": [selected_agent],
        "message": result.decision.reason,
    }
    assert selected_agent in result.decision.reason
    assert service.specialist_call_counts() == {}


@pytest.mark.asyncio
async def test_plan_required_without_executable_workstream_returns_structured_error() -> None:
    # Arrange
    service = OrchestratorService(
        slm_client=RecordingSlmIntentClient(
            IntentSuggestion(intent="unknown", confidence=0.5)
        ),
        intent_classifier=RecordingIntentClassifier(
            LlmIntentAssessment(
                intents=["unknown"],
                confidence=0.9,
                complexity="complex",
                required_agents=["synthesis"],
                rationale="Only synthesis was selected.",
            )
        ),
    )

    # Act
    result = await service.handle_user_request("Summarize without any workstream.")

    # Assert
    assert result.path == "clarification_required"
    assert result.decision.path == "clarification_required"
    assert result.context.decision.path == "clarification_required"
    assert result.context.draft_plan_id is None
    assert result.approval_plan is None
    assert result.specialist_responses == ()
    assert result.a2ui_parts == ()
    assert result.final_artifacts["error"] == {
        "code": "plan_creation_failed",
        "message": result.decision.reason,
    }
    assert "non-synthesis specialist workstream" in result.decision.reason
    assert service.specialist_call_counts() == {}


@pytest.mark.asyncio
async def test_approval_replace_agent_uses_live_registry_after_reload(
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
            "legacy.agent",
            "synthesis",
        ],
    )
    registry = AgentRegistry.from_config_path(config_path)
    service = OrchestratorService(registry=registry)
    proposed = await service.handle_user_request(
        "Prepare me for tomorrow's meeting with ABC Manufacturing."
    )
    assert proposed.approval_plan is not None
    first_step_id = proposed.approval_plan.steps[0].step_id
    surface_id = proposed.approval_plan.approval_surface_id or ""

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

    # Act / Assert
    with pytest.raises(PlanMutationError, match="replacement agent is unavailable"):
        await service.handle_user_action(
            _replace_agent_event(
                proposed.approval_plan.plan_id,
                surface_id,
                step_id=first_step_id,
                replacement_agent_id="legacy.agent",
            )
        )

    edited = await service.handle_user_action(
        _replace_agent_event(
            proposed.approval_plan.plan_id,
            surface_id,
            step_id=first_step_id,
            replacement_agent_id="credit_risk",
        )
    )

    assert edited.status == "draft_updated"
    record = service.approval_record(proposed.approval_plan.plan_id)
    assert record.draft_plan.plan_version == 2
    assert record.draft_plan.steps[0].agent_id == "credit_risk"


@pytest.mark.asyncio
async def test_approval_replace_agent_rejects_descriptor_without_handler_after_reload(
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
    first_step_id = proposed.approval_plan.steps[0].step_id
    surface_id = proposed.approval_plan.approval_surface_id or ""

    _write_registry_config(
        config_path,
        [
            "relationship_summary",
            "internal_knowledge",
            "industry_research",
            "remote_agent",
            "synthesis",
        ],
    )
    registry.reload()

    # Act / Assert
    with pytest.raises(PlanMutationError, match="replacement agent is unavailable"):
        await service.handle_user_action(
            _replace_agent_event(
                proposed.approval_plan.plan_id,
                surface_id,
                step_id=first_step_id,
                replacement_agent_id="remote_agent",
            )
        )

    record = service.approval_record(proposed.approval_plan.plan_id)
    assert record.status == "draft"
    assert record.draft_plan.plan_version == 1
    assert record.draft_plan.steps[0].agent_id != "remote_agent"
    assert service.specialist_call_counts() == {}


@pytest.mark.asyncio
async def test_approval_rejects_stale_draft_agent_removed_after_registry_reload(
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
    assert "industry_research" in proposed.approval_plan.selected_agents

    _write_registry_config(
        config_path,
        [
            "relationship_summary",
            "internal_knowledge",
            "synthesis",
        ],
    )
    registry.reload()

    # Act / Assert
    with pytest.raises(
        PlanMutationError,
        match="approved plan references unavailable agents: industry_research",
    ):
        await service.handle_user_action(
            _approve_event(
                proposed.approval_plan.plan_id,
                proposed.approval_plan.approval_surface_id or "",
                [step.step_id for step in proposed.approval_plan.steps],
            )
        )

    record = service.approval_record(proposed.approval_plan.plan_id)
    assert record.status == "draft"
    assert record.approved_plan is None
    assert service.specialist_call_counts() == {}


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


@pytest.mark.asyncio
async def test_forwarded_a2ui_surfaces_register_to_invoked_owner_not_response_agent() -> None:
    # Arrange
    service = OrchestratorService(
        specialist_user_action_adapters={
            "product_opportunity": MisreportingA2uiUserActionAdapter()
        }
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
                "payload": {"action": "show_detail"},
            }
        }
    )

    # Assert
    assert routed.status == "forwarded"
    assert routed.specialist_responses[0].agent_id == "internal_knowledge"
    owner = service.surface_owner("surface_product_opportunity_detail")
    assert owner is not None
    assert owner.owner_id == "product_opportunity"
