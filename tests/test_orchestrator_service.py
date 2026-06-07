from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from orchestrator_demo.agents import build_default_specialists
from orchestrator_demo.a2a_support.transport import DataPart
from orchestrator_demo.a2ui_support.schema_manager import (
    A2UI_VERSION,
    BASIC_CATALOG_ID,
)
from orchestrator_demo.contracts import (
    AgentDescriptor,
    ExecutionPlan,
    IntentSuggestion,
    LlmIntentAssessment,
    PlanStep,
    SpecialistRequest,
    SpecialistResponse,
)
from orchestrator_demo.orchestrator.graph_runtime import (
    GraphExecutionResult,
    GraphRuntimeError,
    build_graph_spec,
)
from orchestrator_demo.orchestrator.service import OrchestratorService
from orchestrator_demo.orchestrator.surface_routes import SurfaceRouteRegistry


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


class CamelCaseA2uiUserActionAdapter:
    async def handle_user_action(self, _user_action: Any) -> dict[str, Any]:
        surface_id = "surface_product_opportunity_detail"
        return {
            "response_id": "response_product_opportunity_user_action_detail",
            "agent_id": "product_opportunity",
            "content": "Product Opportunity Agent: detail surface ready.",
            "structured_output": {"status": "handled"},
            "a2uiPayload": [
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
                                "component": "Text",
                                "id": "root",
                                "text": "More product detail.",
                            }
                        ],
                    },
                },
            ],
            "surfaceId": surface_id,
        }


class MixedValidityA2uiUserActionAdapter:
    async def handle_user_action(self, user_action: Any) -> SpecialistResponse:
        original_surface_id = user_action["userAction"]["surfaceId"]
        detail_surface_id = "surface_product_opportunity_mixed_detail"
        return SpecialistResponse(
            response_id="response_product_opportunity_mixed_invalid_a2ui",
            agent_id="product_opportunity",
            content="Product Opportunity Agent: mixed follow-up details.",
            structured_output={"status": "handled"},
            a2ui_payload=[
                {
                    "version": A2UI_VERSION,
                    "deleteSurface": {"surfaceId": original_surface_id},
                },
                {
                    "version": A2UI_VERSION,
                    "createSurface": {
                        "surfaceId": detail_surface_id,
                        "catalogId": BASIC_CATALOG_ID,
                    },
                },
                {
                    "version": A2UI_VERSION,
                    "updateComponents": {
                        "surfaceId": detail_surface_id,
                        "components": [
                            {
                                "id": "root",
                                "component": "Text",
                                "text": "Follow-up details.",
                            }
                        ],
                    },
                },
                {
                    "version": A2UI_VERSION,
                    "updateComponents": {
                        "surfaceId": "surface_invalid_specialist_delta",
                        "components": [],
                    },
                },
            ],
            surface_id=detail_surface_id,
        )


class CustomInsightsSpecialist:
    agent_id = "custom.insights"

    def __init__(self) -> None:
        self.call_count = 0
        self.calls: list[SpecialistRequest] = []

    async def handle(self, request: SpecialistRequest) -> SpecialistResponse:
        self.call_count += 1
        self.calls.append(request)
        return SpecialistResponse(
            response_id=f"response_{request.request_id}",
            agent_id=request.agent_id,
            content="Custom Insights Agent: completed.",
            structured_output={"request_id": request.request_id},
        )


class InvalidA2uiSpecialist:
    def __init__(self, delegate) -> None:
        self._delegate = delegate

    @property
    def agent_id(self) -> str:
        return self._delegate.agent_id

    @property
    def call_count(self) -> int:
        return self._delegate.call_count

    @property
    def calls(self) -> list[SpecialistRequest]:
        return self._delegate.calls

    async def handle(self, request: SpecialistRequest) -> SpecialistResponse:
        response = await self._delegate.handle(request)
        return response.model_copy(
            update={
                "a2ui_payload": [
                    {
                        "version": A2UI_VERSION,
                        "updateComponents": {
                            "surfaceId": "surface_invalid_specialist_delta",
                            "components": [],
                        },
                    }
                ],
                "surface_id": "surface_invalid_specialist_delta",
            }
        )


class OwnershipFailingA2uiSpecialist:
    def __init__(self, delegate) -> None:
        self._delegate = delegate

    @property
    def agent_id(self) -> str:
        return self._delegate.agent_id

    @property
    def call_count(self) -> int:
        return self._delegate.call_count

    @property
    def calls(self) -> list[SpecialistRequest]:
        return self._delegate.calls

    async def handle(self, request: SpecialistRequest) -> SpecialistResponse:
        response = await self._delegate.handle(request)
        surface_id = "surface_plan_specialist_claim"
        return response.model_copy(
            update={
                "a2ui_payload": [
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
                                    "component": "Text",
                                    "id": "root",
                                    "text": "Specialist-owned plan prefix claim.",
                                }
                            ],
                        },
                    },
                ],
                "surface_id": surface_id,
            }
        )


class SpoofedA2uiSpecialist:
    agent_id = "product_opportunity"

    def __init__(
        self,
        *,
        surface_id: str = "surface_product_opportunity_spoofed",
        response_agent_id: str = "internal_knowledge",
    ) -> None:
        self.surface_id = surface_id
        self.response_agent_id = response_agent_id
        self.call_count = 0
        self.calls: list[SpecialistRequest] = []

    async def handle(self, request: SpecialistRequest) -> SpecialistResponse:
        self.call_count += 1
        self.calls.append(request)
        return SpecialistResponse(
            response_id=f"response_{request.request_id}",
            agent_id=self.response_agent_id,
            content="Product Opportunity Agent: completed with A2UI.",
            structured_output={"request_id": request.request_id},
            a2ui_payload=_specialist_a2ui(self.surface_id),
            surface_id=self.surface_id,
        )


class ToggleFailingSpecialist:
    agent_id = "internal_knowledge"

    def __init__(self) -> None:
        self.call_count = 0
        self.calls: list[SpecialistRequest] = []
        self.should_fail = True

    async def handle(self, request: SpecialistRequest) -> SpecialistResponse:
        self.call_count += 1
        self.calls.append(request)
        if self.should_fail:
            raise RuntimeError(
                "Authorization: Bearer sk-or-service-secret for customer account 12345"
            )
        return SpecialistResponse(
            response_id=f"response_{request.request_id}",
            agent_id=self.agent_id,
            content="Internal Knowledge Agent: recovered.",
            structured_output={"status": "recovered"},
        )


class SecretBearingA2uiSpecialist:
    agent_id = "internal_knowledge"

    def __init__(self) -> None:
        self.call_count = 0
        self.calls: list[SpecialistRequest] = []

    async def handle(self, request: SpecialistRequest) -> SpecialistResponse:
        self.call_count += 1
        self.calls.append(request)
        return SpecialistResponse(
            response_id=f"response_{request.request_id}",
            agent_id=self.agent_id,
            content="Internal Knowledge Agent: completed with text fallback.",
            structured_output={"status": "completed"},
            a2ui_payload={
                "version": "v0.9",
                "updateComponents": {
                    "surfaceId": "surface_internal_secret",
                    "components": [
                        {
                            "id": "root",
                            "component": "Text",
                            "text": "Authorization: Bearer sk-or-service-secret",
                        }
                    ],
                },
            },
            surface_id="surface_internal_secret",
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


def _remove_step_event(
    plan_id: str,
    surface_id: str,
    *,
    step_id: str,
    plan_version: int = 1,
) -> dict[str, Any]:
    return {
        "userAction": {
            "type": "remove_step",
            "surfaceId": surface_id,
            "payload": {
                "planId": plan_id,
                "editedPlanVersion": plan_version,
                "stepId": step_id,
            },
        }
    }


def _descriptor_source(agent_id: str) -> str:
    return f"""AgentDescriptor(
        agent_id={agent_id!r},
        display_name={agent_id.replace("_", " ").title()!r},
        capabilities=["business banking support"],
        input_schema={{"type": "object"}},
        output_schema={{"type": "object"}},
        a2ui_catalogs=["basic"],
        routing_examples=["Handle a {agent_id} request."],
        execution_mode="local_llm",
    )"""


def _write_registry_config(path: Path, agent_ids: list[str]) -> None:
    path.write_text(
        "from orchestrator_demo.contracts import AgentDescriptor\n\n"
        "AVAILABLE_AGENTS = [\n"
        + ",\n".join(_descriptor_source(agent_id) for agent_id in agent_ids)
        + "\n]\n",
        encoding="utf-8",
    )


def _specialist_a2ui(surface_id: str) -> list[dict[str, Any]]:
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
                        "component": "Text",
                        "id": "root",
                        "text": "Specialist details.",
                    }
                ],
            },
        },
    ]


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
async def test_direct_route_missing_specialist_handler_returns_clarification(
    tmp_path: Path,
) -> None:
    # Arrange
    from orchestrator_demo.registry.agent_registry import AgentRegistry

    config_path = tmp_path / "agent_config.py"
    _write_registry_config(config_path, ["custom_insights"])
    registry = AgentRegistry.from_config_path(config_path)
    slm_client = RecordingSlmIntentClient(
        IntentSuggestion(intent="internal_knowledge", confidence=0.95)
    )
    intent_classifier = RecordingIntentClassifier(
        LlmIntentAssessment(
            intents=["internal_knowledge"],
            confidence=0.95,
            complexity="simple",
            required_agents=["custom_insights"],
            rationale="Injected route to a registry agent with no local handler.",
        )
    )
    service = OrchestratorService(
        registry=registry,
        slm_client=slm_client,
        intent_classifier=intent_classifier,
    )

    # Act
    result = await service.handle_user_request(
        "Use the custom insights agent for this direct request."
    )

    # Assert
    assert result.path == "clarification_required"
    assert result.decision.path == "clarification_required"
    assert result.context.decision.path == "clarification_required"
    assert result.decision.selected_agent is None
    assert "custom_insights" in result.decision.reason
    assert result.specialist_responses == ()
    assert result.a2ui_parts == ()
    assert service.specialist_call_counts() == {}


@pytest.mark.asyncio
async def test_direct_route_slugs_custom_agent_id_in_specialist_request(
    tmp_path: Path,
) -> None:
    # Arrange
    from orchestrator_demo.registry.agent_registry import AgentRegistry

    config_path = tmp_path / "agent_config.py"
    _write_registry_config(config_path, ["custom.insights"])
    registry = AgentRegistry.from_config_path(config_path)
    specialist = CustomInsightsSpecialist()
    slm_client = RecordingSlmIntentClient(
        IntentSuggestion(intent="internal_knowledge", confidence=0.95)
    )
    intent_classifier = RecordingIntentClassifier(
        LlmIntentAssessment(
            intents=["internal_knowledge"],
            confidence=0.95,
            complexity="simple",
            required_agents=["custom.insights"],
            rationale="Injected direct route to a custom registry agent.",
        )
    )
    service = OrchestratorService(
        registry=registry,
        slm_client=slm_client,
        intent_classifier=intent_classifier,
        specialists={"custom.insights": specialist},
    )

    # Act
    result = await service.handle_user_request(
        "Use the custom insights agent for this direct request."
    )

    # Assert
    assert result.path == "direct"
    assert specialist.call_count == 1
    request = specialist.calls[0]
    assert request.agent_id == "custom.insights"
    assert request.request_id == (
        f"request_direct_{result.context.plan_scope_id}_custom_insights"
    )
    assert "." not in request.request_id
    assert result.specialist_responses[0].structured_output["request_id"] == (
        request.request_id
    )


@pytest.mark.asyncio
async def test_explicit_empty_specialist_map_is_honored_for_direct_routes() -> None:
    # Arrange
    slm_client = RecordingSlmIntentClient(
        IntentSuggestion(intent="internal_knowledge", confidence=0.95)
    )
    intent_classifier = RecordingIntentClassifier(
        LlmIntentAssessment(
            intents=["internal_knowledge"],
            confidence=0.95,
            complexity="simple",
            required_agents=["internal_knowledge"],
            rationale="Injected direct route with no executable handlers.",
        )
    )
    service = OrchestratorService(
        slm_client=slm_client,
        intent_classifier=intent_classifier,
        specialists={},
    )

    # Act
    result = await service.handle_user_request("Summarize internal notes.")

    # Assert
    assert result.path == "clarification_required"
    assert result.decision.selected_agent is None
    assert "internal_knowledge" in result.decision.reason
    assert result.specialist_responses == ()
    assert result.a2ui_parts == ()
    assert service.specialist_call_counts() == {}


@pytest.mark.asyncio
async def test_direct_route_specialist_failure_is_redacted() -> None:
    # Arrange
    failing_specialist = ToggleFailingSpecialist()
    slm_client = RecordingSlmIntentClient(
        IntentSuggestion(intent="internal_knowledge", confidence=0.95)
    )
    intent_classifier = RecordingIntentClassifier(
        LlmIntentAssessment(
            intents=["internal_knowledge"],
            confidence=0.95,
            complexity="simple",
            required_agents=["internal_knowledge"],
            rationale="Injected direct route to a failing specialist.",
        )
    )
    service = OrchestratorService(
        slm_client=slm_client,
        intent_classifier=intent_classifier,
        specialists={"internal_knowledge": failing_specialist},
    )

    # Act / Assert
    with pytest.raises(GraphRuntimeError) as exc_info:
        await service.handle_user_request("Summarize internal notes.")

    error_message = str(exc_info.value)
    assert "RuntimeError. Error details redacted." in error_message
    assert "Authorization" not in error_message
    assert "sk-or-service-secret" not in error_message
    assert failing_specialist.call_count == 1


@pytest.mark.asyncio
async def test_repeated_direct_a2ui_requests_get_unique_response_and_surface_ids() -> None:
    # Arrange
    slm_client = RecordingSlmIntentClient(
        IntentSuggestion(intent="product_opportunity", confidence=0.95)
    )
    intent_classifier = RecordingIntentClassifier(
        LlmIntentAssessment(
            intents=["product_opportunity"],
            confidence=0.95,
            complexity="simple",
            required_agents=["product_opportunity"],
            rationale="Injected single-agent product opportunity assessment.",
        )
    )
    service = OrchestratorService(
        slm_client=slm_client,
        intent_classifier=intent_classifier,
    )
    user_input = "What product opportunities should I consider for a cafe business?"

    # Act
    first = await service.handle_user_request(user_input)
    second = await service.handle_user_request(user_input)

    # Assert
    assert first.path == "direct"
    assert second.path == "direct"
    assert len(first.specialist_responses) == 1
    assert len(second.specialist_responses) == 1
    first_response = first.specialist_responses[0]
    second_response = second.specialist_responses[0]

    assert first_response.response_id != second_response.response_id
    assert first_response.surface_id is not None
    assert second_response.surface_id is not None
    assert first_response.surface_id != second_response.surface_id
    assert first_response.structured_output["request_id"] != (
        second_response.structured_output["request_id"]
    )
    first_owner = service.surface_owner(first_response.surface_id)
    second_owner = service.surface_owner(second_response.surface_id)
    assert first_owner is not None
    assert second_owner is not None
    assert first_owner.owner_id == "product_opportunity"
    assert second_owner.owner_id == "product_opportunity"


@pytest.mark.asyncio
async def test_direct_a2ui_surface_owner_uses_invoked_agent_not_response_agent() -> None:
    # Arrange
    spoofing_specialist = SpoofedA2uiSpecialist()
    slm_client = RecordingSlmIntentClient(
        IntentSuggestion(intent="product_opportunity", confidence=0.95)
    )
    intent_classifier = RecordingIntentClassifier(
        LlmIntentAssessment(
            intents=["product_opportunity"],
            confidence=0.95,
            complexity="simple",
            required_agents=["product_opportunity"],
            rationale="Injected direct route with spoofed response metadata.",
        )
    )
    service = OrchestratorService(
        slm_client=slm_client,
        intent_classifier=intent_classifier,
        specialists={"product_opportunity": spoofing_specialist},
    )

    # Act
    result = await service.handle_user_request(
        "What product opportunities should I consider for a cafe business?"
    )

    # Assert
    assert result.path == "direct"
    assert result.specialist_responses[0].agent_id == "internal_knowledge"
    assert len(result.a2ui_parts) == 2
    owner = service.surface_owner(spoofing_specialist.surface_id)
    assert owner is not None
    assert owner.owner_type == "specialist"
    assert owner.owner_id == "product_opportunity"


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
async def test_complex_request_with_missing_specialist_handler_returns_clarification() -> None:
    # Arrange
    specialists = build_default_specialists()
    internal_knowledge = specialists["internal_knowledge"]
    slm_client, intent_classifier = _complex_internal_knowledge_classifier()
    service = OrchestratorService(
        slm_client=slm_client,
        intent_classifier=intent_classifier,
        specialists={"internal_knowledge": internal_knowledge},
    )

    # Act
    result = await service.handle_user_request(
        "Prepare a focused internal-knowledge workflow."
    )

    # Assert
    assert result.path == "clarification_required"
    assert result.approval_plan is None
    assert result.specialist_responses == ()
    assert result.a2ui_parts == ()
    assert "synthesis" in result.decision.reason
    assert service.specialist_call_counts() == {}


@pytest.mark.asyncio
async def test_unformable_plan_returns_clarification_instead_of_planner_error() -> None:
    # Arrange
    slm_client = RecordingSlmIntentClient(
        IntentSuggestion(intent="meeting_prep", confidence=0.9)
    )
    intent_classifier = RecordingIntentClassifier(
        LlmIntentAssessment(
            intents=["meeting_prep"],
            confidence=0.94,
            complexity="complex",
            required_agents=["synthesis"],
            rationale="Injected synthesis-only plan assessment.",
        )
    )
    service = OrchestratorService(
        slm_client=slm_client,
        intent_classifier=intent_classifier,
    )

    # Act
    result = await service.handle_user_request(
        "Synthesize the answer without any source workstream."
    )

    # Assert
    assert result.path == "clarification_required"
    assert result.decision.path == "clarification_required"
    assert result.context.decision.path == "clarification_required"
    assert "no available non-synthesis specialist workstream" in result.decision.reason
    assert result.approval_plan is None
    assert result.specialist_responses == ()
    assert result.a2ui_parts == ()
    assert service.specialist_call_counts() == {}


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


@pytest.mark.parametrize("final_action", ["approved", "rejected"])
@pytest.mark.asyncio
async def test_eval_mode_repeated_final_plan_prompt_gets_new_draft_id(
    monkeypatch: pytest.MonkeyPatch,
    final_action: str,
) -> None:
    # Arrange
    monkeypatch.setenv("ORCHESTRATOR_DEMO_DETERMINISTIC_MODEL", "1")
    monkeypatch.setenv("ORCHESTRATOR_DEMO_ADK_EVAL_MODE", "1")
    service = OrchestratorService()
    user_input = "Prepare me for a meeting with ABC Manufacturing."
    first = await service.handle_user_request(user_input)
    assert first.approval_plan is not None
    first_plan = first.approval_plan

    if final_action == "approved":
        final_event = _approve_event(
            first_plan.plan_id,
            first_plan.approval_surface_id or "",
            [step.step_id for step in first_plan.steps],
        )
    else:
        final_event = _reject_event(
            first_plan.plan_id,
            first_plan.approval_surface_id or "",
        )
    finalized = await service.handle_user_action(final_event)
    assert finalized.status == final_action

    # Act
    second = await service.handle_user_request(user_input)

    # Assert
    assert second.path == "plan_required"
    assert second.approval_plan is not None
    assert second.context.plan_scope_id == f"{first.context.plan_scope_id}_2"
    assert second.approval_plan.plan_id == f"{first_plan.plan_id}_2"
    assert service.approval_record(first_plan.plan_id).status == final_action
    assert service.approval_record(second.approval_plan.plan_id).status == "draft"


@pytest.mark.asyncio
async def test_eval_mode_repeated_pending_plan_prompt_gets_new_draft_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    monkeypatch.setenv("ORCHESTRATOR_DEMO_DETERMINISTIC_MODEL", "1")
    monkeypatch.setenv("ORCHESTRATOR_DEMO_ADK_EVAL_MODE", "1")
    service = OrchestratorService()
    user_input = "Prepare me for a meeting with ABC Manufacturing."
    first = await service.handle_user_request(user_input)
    assert first.approval_plan is not None
    first_plan = first.approval_plan
    first_step_id = first_plan.steps[0].step_id

    edited = await service.handle_user_action(
        _add_instruction_event(
            first_plan.plan_id,
            first_plan.approval_surface_id or "",
            step_id=first_step_id,
        )
    )
    assert edited.status == "draft_updated"

    # Act
    second = await service.handle_user_request(user_input)

    # Assert
    assert second.path == "plan_required"
    assert second.approval_plan is not None
    assert second.context.plan_scope_id == f"{first.context.plan_scope_id}_2"
    assert second.approval_plan.plan_id == f"{first_plan.plan_id}_2"

    first_record = service.approval_record(first_plan.plan_id)
    assert first_record.status == "draft"
    assert first_record.draft_plan.plan_version == 2
    assert "Additional instruction: Prioritize covenant follow-ups." in (
        first_record.draft_plan.steps[0].instruction
    )
    assert service.approval_record(second.approval_plan.plan_id).status == "draft"


@pytest.mark.asyncio
async def test_approval_returns_graph_result_when_specialist_a2ui_is_invalid() -> None:
    # Arrange
    invalid_a2ui_specialist = SecretBearingA2uiSpecialist()
    specialists = build_default_specialists()
    specialists["internal_knowledge"] = invalid_a2ui_specialist
    slm_client, intent_classifier = _complex_internal_knowledge_classifier()
    service = OrchestratorService(
        slm_client=slm_client,
        intent_classifier=intent_classifier,
        specialists=specialists,
    )
    proposed = await service.handle_user_request(
        "Prepare a focused internal-knowledge workflow."
    )
    assert proposed.approval_plan is not None
    plan_id = proposed.approval_plan.plan_id
    surface_id = proposed.approval_plan.approval_surface_id or ""
    step_ids = [step.step_id for step in proposed.approval_plan.steps]

    # Act
    result = await service.handle_user_action(
        _approve_event(plan_id, surface_id, step_ids)
    )

    # Assert
    assert result.status == "approved"
    assert result.graph_execution is not None
    assert [response.agent_id for response in result.specialist_responses] == [
        "internal_knowledge",
        "synthesis",
    ]
    fallback_response = result.specialist_responses[0]
    assert fallback_response.a2ui_payload is not None
    assert fallback_response.surface_id is not None
    assert fallback_response.surface_id.startswith("surface_fallback_response_")
    assert any(
        part.data.get("deleteSurface", {}).get("surfaceId") == surface_id
        for part in result.a2ui_parts
    )
    assert any(
        part.data.get("updateComponents", {}).get("surfaceId")
        == fallback_response.surface_id
        for part in result.a2ui_parts
    )
    assert result.final_artifacts["final_response"].agent_id == "synthesis"
    record = service.approval_record(plan_id)
    assert record.status == "approved"
    assert record.approved_plan is not None
    assert invalid_a2ui_specialist.call_count == 1
    assert service.surface_owner("surface_internal_secret") is None


@pytest.mark.asyncio
async def test_graph_a2ui_surface_owner_uses_invoked_step_agent_not_response_agent() -> None:
    # Arrange
    spoofing_specialist = SpoofedA2uiSpecialist(
        surface_id="surface_internal_graph_spoofed",
        response_agent_id="product_opportunity",
    )
    specialists = build_default_specialists()
    specialists["internal_knowledge"] = spoofing_specialist
    slm_client, intent_classifier = _complex_internal_knowledge_classifier()
    service = OrchestratorService(
        slm_client=slm_client,
        intent_classifier=intent_classifier,
        specialists=specialists,
    )
    proposed = await service.handle_user_request(
        "Prepare a focused internal-knowledge workflow."
    )
    assert proposed.approval_plan is not None

    # Act
    result = await service.handle_user_action(
        _approve_event(
            proposed.approval_plan.plan_id,
            proposed.approval_plan.approval_surface_id or "",
            [step.step_id for step in proposed.approval_plan.steps],
        )
    )

    # Assert
    assert result.status == "approved"
    assert result.graph_execution is not None
    assert result.specialist_responses[0].agent_id == "product_opportunity"
    specialist_parts = [
        part for part in result.a2ui_parts if "deleteSurface" not in part.data
    ]
    assert len(specialist_parts) == 2
    owner = service.surface_owner(spoofing_specialist.surface_id)
    assert owner is not None
    assert owner.owner_type == "specialist"
    assert owner.owner_id == "internal_knowledge"


def test_graph_a2ui_surface_owner_uses_response_request_completion_order() -> None:
    # Arrange
    service = OrchestratorService()
    plan = ExecutionPlan(
        plan_id="plan_parallel_owner_registration",
        objective="Prepare parallel context.",
        detected_intents=["meeting_prep"],
        selected_agents=[
            "relationship_summary",
            "internal_knowledge",
            "synthesis",
        ],
        steps=[
            PlanStep(
                step_id="step_relationship_summary",
                agent_id="relationship_summary",
                instruction="Summarize relationship history.",
                expected_output="Relationship context.",
                parallel_group="parallel_context",
            ),
            PlanStep(
                step_id="step_internal_knowledge",
                agent_id="internal_knowledge",
                instruction="Review internal notes.",
                expected_output="Internal context.",
                parallel_group="parallel_context",
            ),
            PlanStep(
                step_id="step_synthesis",
                agent_id="synthesis",
                instruction="Synthesize the brief.",
                expected_output="Final brief.",
                depends_on=[
                    "step_relationship_summary",
                    "step_internal_knowledge",
                ],
            ),
        ],
        approval_surface_id="surface_plan_parallel_owner_registration",
    )
    relationship_request = SpecialistRequest(
        request_id="request_parallel_relationship",
        user_input="Summarize relationship history.",
        agent_id="relationship_summary",
        plan_id=plan.plan_id,
        step_id="step_relationship_summary",
    )
    internal_request = SpecialistRequest(
        request_id="request_parallel_internal",
        user_input="Review internal notes.",
        agent_id="internal_knowledge",
        plan_id=plan.plan_id,
        step_id="step_internal_knowledge",
    )
    relationship_response = SpecialistResponse(
        response_id="response_parallel_relationship",
        agent_id="relationship_summary",
        content="Relationship Summary Agent: completed with A2UI.",
        a2ui_payload=_specialist_a2ui("surface_relationship_parallel"),
        surface_id="surface_relationship_parallel",
    )
    internal_response = SpecialistResponse(
        response_id="response_parallel_internal",
        agent_id="internal_knowledge",
        content="Internal Knowledge Agent: completed with A2UI.",
        a2ui_payload=_specialist_a2ui("surface_internal_parallel"),
        surface_id="surface_internal_parallel",
    )
    graph_execution = GraphExecutionResult(
        graph=build_graph_spec(plan),
        workflow=object(),
        status_events=(),
        specialist_requests=(relationship_request, internal_request),
        specialist_responses=(internal_response, relationship_response),
        specialist_response_requests=(internal_request, relationship_request),
        adk_event_outputs=(),
    )

    # Act
    parts = service._prepare_graph_response_a2ui(graph_execution)

    # Assert
    assert len(parts) == 4
    internal_owner = service.surface_owner("surface_internal_parallel")
    relationship_owner = service.surface_owner("surface_relationship_parallel")
    assert internal_owner is not None
    assert relationship_owner is not None
    assert internal_owner.owner_id == "internal_knowledge"
    assert relationship_owner.owner_id == "relationship_summary"


@pytest.mark.asyncio
async def test_failed_graph_execution_resets_request_approval_for_recovery() -> None:
    # Arrange
    from orchestrator_demo.orchestrator.graph_runtime import GraphRuntimeError

    failing_specialist = ToggleFailingSpecialist()
    specialists = build_default_specialists()
    specialists["internal_knowledge"] = failing_specialist
    slm_client, intent_classifier = _complex_internal_knowledge_classifier()
    service = OrchestratorService(
        slm_client=slm_client,
        intent_classifier=intent_classifier,
        specialists=specialists,
    )
    proposed = await service.handle_user_request(
        "Prepare a focused internal-knowledge workflow."
    )
    assert proposed.approval_plan is not None
    plan_id = proposed.approval_plan.plan_id
    surface_id = proposed.approval_plan.approval_surface_id or ""
    step_ids = [step.step_id for step in proposed.approval_plan.steps]

    # Act / Assert
    with pytest.raises(GraphRuntimeError) as exc_info:
        await service.handle_user_action(_approve_event(plan_id, surface_id, step_ids))

    assert "Authorization: Bearer" not in str(exc_info.value)
    failed_record = service.approval_record(plan_id)
    assert failed_record.status == "draft"
    assert failed_record.approved_plan is None

    edited = await service.handle_user_action(
        _add_instruction_event(
            plan_id,
            surface_id,
            step_id="step_internal_knowledge",
        )
    )
    assert edited.status == "draft_updated"
    assert service.approval_record(plan_id).draft_plan.plan_version == 2

    failing_specialist.should_fail = False
    approved = await service.handle_user_action(
        _approve_event(plan_id, surface_id, step_ids, plan_version=2)
    )

    assert approved.status == "approved"
    assert approved.graph_execution is not None
    assert service.approval_record(plan_id).status == "approved"
    assert failing_specialist.call_count == 2


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
async def test_replace_agent_uses_live_registry_after_reload(
    tmp_path: Path,
) -> None:
    # Arrange
    from orchestrator_demo.orchestrator.approval_state import PlanMutationError
    from orchestrator_demo.registry.agent_registry import AgentRegistry

    config_path = tmp_path / "agent_config.py"
    _write_registry_config(
        config_path,
        ["internal_knowledge", "credit_risk", "synthesis"],
    )
    registry = AgentRegistry.from_config_path(config_path)
    slm_client, intent_classifier = _complex_internal_knowledge_classifier()
    service = OrchestratorService(
        registry=registry,
        slm_client=slm_client,
        intent_classifier=intent_classifier,
    )
    proposed = await service.handle_user_request(
        "Prepare a focused internal-knowledge workflow."
    )
    assert proposed.approval_plan is not None
    plan_id = proposed.approval_plan.plan_id
    surface_id = proposed.approval_plan.approval_surface_id or ""

    _write_registry_config(config_path, ["internal_knowledge", "synthesis"])
    registry.reload()

    # Act / Assert
    with pytest.raises(PlanMutationError, match="replacement agent is unavailable"):
        await service.handle_user_action(
            _replace_agent_event(
                plan_id,
                surface_id,
                step_id="step_internal_knowledge",
                replacement_agent_id="credit_risk",
            )
        )

    record = service.approval_record(plan_id)
    assert record.status == "draft"
    assert record.draft_plan.plan_version == 1
    assert record.draft_plan.selected_agents == [
        "internal_knowledge",
        "synthesis",
    ]


@pytest.mark.asyncio
async def test_replace_agent_rejects_descriptor_without_specialist_handler(
    tmp_path: Path,
) -> None:
    # Arrange
    from orchestrator_demo.orchestrator.approval_state import PlanMutationError
    from orchestrator_demo.registry.agent_registry import AgentRegistry

    config_path = tmp_path / "agent_config.py"
    _write_registry_config(
        config_path,
        ["internal_knowledge", "custom_insights", "synthesis"],
    )
    registry = AgentRegistry.from_config_path(config_path)
    slm_client, intent_classifier = _complex_internal_knowledge_classifier()
    service = OrchestratorService(
        registry=registry,
        slm_client=slm_client,
        intent_classifier=intent_classifier,
    )
    proposed = await service.handle_user_request(
        "Prepare a focused internal-knowledge workflow."
    )
    assert proposed.approval_plan is not None
    plan_id = proposed.approval_plan.plan_id
    surface_id = proposed.approval_plan.approval_surface_id or ""

    # Act / Assert
    with pytest.raises(
        PlanMutationError,
        match="agents without executable handlers: custom_insights",
    ):
        await service.handle_user_action(
            _replace_agent_event(
                plan_id,
                surface_id,
                step_id="step_internal_knowledge",
                replacement_agent_id="custom_insights",
            )
        )

    record = service.approval_record(plan_id)
    assert record.status == "draft"
    assert record.draft_plan.plan_version == 1
    assert record.draft_plan.selected_agents == [
        "internal_knowledge",
        "synthesis",
    ]
    assert service.specialist_call_counts() == {}


@pytest.mark.asyncio
async def test_draft_can_repair_multiple_unavailable_agents_after_reload(
    tmp_path: Path,
) -> None:
    # Arrange
    from orchestrator_demo.registry.agent_registry import AgentRegistry

    config_path = tmp_path / "agent_config.py"
    _write_registry_config(
        config_path,
        ["internal_knowledge", "credit_risk", "product_opportunity", "synthesis"],
    )
    registry = AgentRegistry.from_config_path(config_path)
    slm_client = RecordingSlmIntentClient(
        IntentSuggestion(intent="meeting_prep", confidence=0.9)
    )
    intent_classifier = RecordingIntentClassifier(
        LlmIntentAssessment(
            intents=["meeting_prep"],
            confidence=0.94,
            complexity="complex",
            required_agents=["internal_knowledge", "credit_risk", "synthesis"],
            rationale="Injected two unavailable workstreams for repair.",
        )
    )
    service = OrchestratorService(
        registry=registry,
        slm_client=slm_client,
        intent_classifier=intent_classifier,
    )
    proposed = await service.handle_user_request(
        "Prepare a meeting brief with internal notes and credit risk."
    )
    assert proposed.approval_plan is not None
    plan_id = proposed.approval_plan.plan_id
    surface_id = proposed.approval_plan.approval_surface_id or ""
    assert proposed.approval_plan.selected_agents == [
        "internal_knowledge",
        "credit_risk",
        "synthesis",
    ]

    _write_registry_config(config_path, ["product_opportunity", "synthesis"])
    registry.reload()

    # Act
    replaced = await service.handle_user_action(
        _replace_agent_event(
            plan_id,
            surface_id,
            step_id="step_internal_knowledge",
            replacement_agent_id="product_opportunity",
        )
    )
    removed = await service.handle_user_action(
        _remove_step_event(
            plan_id,
            surface_id,
            step_id="step_credit_risk",
            plan_version=2,
        )
    )
    repaired_record = service.approval_record(plan_id)
    approved = await service.handle_user_action(
        _approve_event(
            plan_id,
            surface_id,
            [step.step_id for step in repaired_record.draft_plan.steps],
            plan_version=3,
        )
    )

    # Assert
    assert replaced.status == "draft_updated"
    assert removed.status == "draft_updated"
    assert repaired_record.draft_plan.plan_version == 3
    assert repaired_record.draft_plan.selected_agents == [
        "product_opportunity",
        "synthesis",
    ]
    assert [step.agent_id for step in repaired_record.draft_plan.steps] == [
        "product_opportunity",
        "synthesis",
    ]
    assert approved.status == "approved"
    assert approved.graph_execution is not None
    assert service.specialist_call_counts() == {
        "product_opportunity": 1,
        "synthesis": 1,
    }


@pytest.mark.asyncio
async def test_approve_plan_uses_live_registry_after_reload(
    tmp_path: Path,
) -> None:
    # Arrange
    from orchestrator_demo.orchestrator.approval_state import PlanMutationError
    from orchestrator_demo.registry.agent_registry import AgentRegistry

    config_path = tmp_path / "agent_config.py"
    _write_registry_config(config_path, ["internal_knowledge", "synthesis"])
    registry = AgentRegistry.from_config_path(config_path)
    slm_client, intent_classifier = _complex_internal_knowledge_classifier()
    service = OrchestratorService(
        registry=registry,
        slm_client=slm_client,
        intent_classifier=intent_classifier,
    )
    proposed = await service.handle_user_request(
        "Prepare a focused internal-knowledge workflow."
    )
    assert proposed.approval_plan is not None
    plan_id = proposed.approval_plan.plan_id
    surface_id = proposed.approval_plan.approval_surface_id or ""
    step_ids = [step.step_id for step in proposed.approval_plan.steps]

    _write_registry_config(config_path, ["synthesis"])
    registry.reload()

    # Act / Assert
    with pytest.raises(
        PlanMutationError,
        match="plan references unavailable agents: internal_knowledge",
    ):
        await service.handle_user_action(
            _approve_event(plan_id, surface_id, step_ids)
        )

    record = service.approval_record(plan_id)
    assert record.status == "draft"
    assert record.approved_plan is None
    assert service.specialist_call_counts() == {}


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


@pytest.mark.asyncio
async def test_remote_wrapper_user_action_handler_is_used_by_default() -> None:
    # Arrange
    from orchestrator_demo.a2a_support.local_remote_wrapper import LocalRemoteAgentWrapper
    from orchestrator_demo.agents.product_opportunity import ProductOpportunityAgent

    local_agent = ProductOpportunityAgent()
    wrapper = LocalRemoteAgentWrapper(local_agent)
    surface_registry = SurfaceRouteRegistry()
    surface_id = "surface_product_opportunity_request_product_opportunity"
    surface_registry.register_specialist_surface(
        surface_id,
        agent_id=wrapper.agent_id,
    )
    service = OrchestratorService(
        specialists={wrapper.agent_id: wrapper},
        surface_registry=surface_registry,
    )
    user_action = {
        "userAction": {
            "type": "specialist_action",
            "surfaceId": surface_id,
            "payload": {
                "selectedProduct": "treasury_services",
                "filters": ["cash_visibility", "controls"],
            },
        }
    }

    # Act
    routed = await service.handle_user_action(user_action)

    # Assert
    assert routed.status == "forwarded"
    assert local_agent.call_count == 1
    assert local_agent.calls[0].context["user_action_payload"] == user_action
    assert len(routed.specialist_responses) == 1
    response = routed.specialist_responses[0]
    assert response.agent_id == "product_opportunity"
    assert response.response_id.startswith("response_product_opportunity_request_")


@pytest.mark.asyncio
async def test_default_specialist_user_action_adapter_slugs_custom_agent_response_id() -> None:
    # Arrange
    surface_registry = SurfaceRouteRegistry()
    surface_registry.register_specialist_surface(
        "surface_custom_insights_action",
        agent_id="custom.insights",
    )
    service = OrchestratorService(
        specialists={"custom.insights": CustomInsightsSpecialist()},
        surface_registry=surface_registry,
    )

    # Act
    routed = await service.handle_user_action(
        {
            "userAction": {
                "type": "specialist_action",
                "surfaceId": "surface_custom_insights_action",
                "payload": {"action": "show_more_detail"},
            }
        }
    )

    # Assert
    assert routed.status == "forwarded"
    assert routed.surface_route_result is not None
    assert routed.surface_route_result.owner is not None
    assert routed.surface_route_result.owner.owner_id == "custom.insights"
    assert len(routed.specialist_responses) == 1
    response = routed.specialist_responses[0]
    assert response.agent_id == "custom.insights"
    assert response.response_id == "response_custom_insights_user_action"
    assert "." not in response.response_id


@pytest.mark.asyncio
async def test_invalid_graph_specialist_a2ui_returns_valid_fallback_after_approval() -> (
    None
):
    specialists = build_default_specialists()
    specialists["relationship_summary"] = InvalidA2uiSpecialist(
        specialists["relationship_summary"]
    )
    service = OrchestratorService(specialists=specialists)
    proposed = await service.handle_user_request(
        "Prepare me for tomorrow's meeting with ABC Manufacturing."
    )
    assert proposed.approval_plan is not None
    plan = proposed.approval_plan

    approved = await service.handle_user_action(
        _approve_event(
            plan.plan_id,
            plan.approval_surface_id or "",
            [step.step_id for step in plan.steps],
        )
    )

    assert approved.status == "approved"
    assert service.approval_record(plan.plan_id).status == "approved"
    fallback_response = next(
        response
        for response in approved.specialist_responses
        if response.agent_id == "relationship_summary"
    )
    assert fallback_response.a2ui_payload is not None
    assert fallback_response.surface_id is not None
    assert fallback_response.surface_id.startswith("surface_fallback_response_")
    assert all(
        part.data.get("updateComponents", {}).get("surfaceId")
        != "surface_invalid_specialist_delta"
        for part in approved.a2ui_parts
    )
    assert any(
        part.data.get("updateComponents", {}).get("surfaceId")
        == fallback_response.surface_id
        for part in approved.a2ui_parts
    )


@pytest.mark.asyncio
async def test_specialist_a2ui_ownership_failure_returns_fallback_after_approval() -> (
    None
):
    specialists = build_default_specialists()
    specialists["relationship_summary"] = OwnershipFailingA2uiSpecialist(
        specialists["relationship_summary"]
    )
    service = OrchestratorService(specialists=specialists)
    proposed = await service.handle_user_request(
        "Prepare me for tomorrow's meeting with ABC Manufacturing."
    )
    assert proposed.approval_plan is not None
    plan = proposed.approval_plan

    approved = await service.handle_user_action(
        _approve_event(
            plan.plan_id,
            plan.approval_surface_id or "",
            [step.step_id for step in plan.steps],
        )
    )

    assert approved.status == "approved"
    assert service.approval_record(plan.plan_id).status == "approved"
    fallback_response = next(
        response
        for response in approved.specialist_responses
        if response.agent_id == "relationship_summary"
    )
    assert fallback_response.surface_id is not None
    assert fallback_response.surface_id.startswith("surface_fallback_response_")
    assert all(
        part.data.get("createSurface", {}).get("surfaceId")
        != "surface_plan_specialist_claim"
        for part in approved.a2ui_parts
    )
    assert any(
        part.data.get("createSurface", {}).get("surfaceId")
        == fallback_response.surface_id
        for part in approved.a2ui_parts
    )


@pytest.mark.asyncio
async def test_specialist_user_action_response_normalizes_camel_case_a2ui() -> None:
    service = OrchestratorService(
        specialist_user_action_adapters={
            "product_opportunity": CamelCaseA2uiUserActionAdapter()
        }
    )
    result = await service.handle_user_request(
        "What product opportunities should I consider for a cafe business?"
    )
    surface_id = result.specialist_responses[0].surface_id
    assert surface_id is not None

    routed = await service.handle_user_action(
        {
            "userAction": {
                "type": "specialist_action",
                "surfaceId": surface_id,
                "payload": {"action": "show_more_detail"},
            }
        }
    )

    assert routed.status == "forwarded"
    assert [response.agent_id for response in routed.specialist_responses] == [
        "product_opportunity"
    ]
    assert len(routed.a2ui_parts) == 2
    assert routed.final_artifacts["final_response"].surface_id == (
        "surface_product_opportunity_detail"
    )
    assert service.surface_owner("surface_product_opportunity_detail") is not None


@pytest.mark.asyncio
async def test_specialist_user_action_invalid_a2ui_fallback_does_not_keep_stale_owner() -> (
    None
):
    service = OrchestratorService(
        specialist_user_action_adapters={
            "product_opportunity": MixedValidityA2uiUserActionAdapter()
        }
    )
    result = await service.handle_user_request(
        "What product opportunities should I consider for a cafe business?"
    )
    surface_id = result.specialist_responses[0].surface_id
    assert surface_id is not None
    original_owner = service.surface_owner(surface_id)
    assert original_owner is not None

    routed = await service.handle_user_action(
        {
            "userAction": {
                "type": "specialist_action",
                "surfaceId": surface_id,
                "payload": {"action": "show_more_detail"},
            }
        }
    )

    assert routed.status == "forwarded"
    fallback_response = routed.final_artifacts["final_response"]
    assert fallback_response.surface_id is not None
    assert fallback_response.surface_id.startswith("surface_fallback_response_")
    assert service.surface_owner(surface_id) == original_owner
    assert service.surface_owner("surface_product_opportunity_mixed_detail") is None
    assert service.surface_owner("surface_invalid_specialist_delta") is None
    fallback_owner = service.surface_owner(fallback_response.surface_id)
    assert fallback_owner is not None
    assert fallback_owner.owner_id == "product_opportunity"
    assert all(
        part.data.get("createSurface", {}).get("surfaceId")
        != "surface_product_opportunity_mixed_detail"
        for part in routed.a2ui_parts
    )
    assert any(
        part.data.get("createSurface", {}).get("surfaceId")
        == fallback_response.surface_id
        for part in routed.a2ui_parts
    )
