import json
from pathlib import Path
from typing import Any

import pytest

from orchestrator_demo.contracts import (
    AgentDescriptor,
    IntentSuggestion,
    LlmIntentAssessment,
    RoutingDecision,
    SpecialistRequest,
    SpecialistResponse,
)
from orchestrator_demo.intent.classifier import DeterministicIntentClassifier
from orchestrator_demo.intent.slm_mock_client import MockSlmIntentClient
from orchestrator_demo.orchestrator.request_context import (
    RequestContext,
    SpecialistPreApprovalError,
    call_specialist_with_guard,
)
from orchestrator_demo.orchestrator.planner import DraftExecutionPlanner
from orchestrator_demo.orchestrator.router import RequestRouter
from orchestrator_demo.registry.agent_registry import AgentRegistry


GOLDEN_SCENARIOS_PATH = Path(__file__).with_name("golden_scenarios.json")


def _golden_plan_cases() -> list[dict[str, Any]]:
    return [
        case
        for case in json.loads(GOLDEN_SCENARIOS_PATH.read_text())
        if case["expected_path"] == "plan_required"
    ]


class _RecordingSlmClient:
    def __init__(self, suggestion: IntentSuggestion) -> None:
        self._suggestion = suggestion
        self.calls: list[str] = []

    async def classify(self, user_input: str) -> IntentSuggestion:
        self.calls.append(user_input)
        return self._suggestion


class _RecordingClassifier:
    def __init__(
        self,
        assessment: LlmIntentAssessment,
        slm_client: _RecordingSlmClient,
    ) -> None:
        self._assessment = assessment
        self._slm_client = slm_client
        self.assess_calls: list[tuple[str, IntentSuggestion]] = []
        self.slm_call_counts_seen: list[int] = []

    async def assess(
        self,
        user_input: str,
        slm_suggestion: IntentSuggestion,
        *,
        available_agents: object | None = None,
    ) -> LlmIntentAssessment:
        del available_agents
        self.assess_calls.append((user_input, slm_suggestion))
        self.slm_call_counts_seen.append(len(self._slm_client.calls))
        return self._assessment


class _FakeSpecialist:
    def __init__(self) -> None:
        self.calls: list[SpecialistRequest] = []

    async def __call__(self, request: SpecialistRequest) -> SpecialistResponse:
        self.calls.append(request)
        return SpecialistResponse(
            response_id="response_fake",
            agent_id=request.agent_id,
            content="fake specialist response",
        )


class _StaticRegistry:
    def __init__(self, descriptors: list[AgentDescriptor]) -> None:
        self._descriptors = descriptors

    def descriptors(self) -> list[AgentDescriptor]:
        return list(self._descriptors)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", _golden_plan_cases(), ids=lambda case: case["name"])
async def test_golden_complex_routes_generate_complete_draft_plans(
    case: dict[str, Any],
) -> None:
    # Arrange
    registry = AgentRegistry.from_default_config()
    router = RequestRouter(
        slm_client=MockSlmIntentClient(),
        intent_classifier=DeterministicIntentClassifier(),
        registry=registry,
    )
    planner = DraftExecutionPlanner(registry=registry)

    # Act
    context = await router.route_request(case["input"])
    plan = planner.create_plan(context)

    # Assert
    assert context.decision.path == "plan_required"
    assert plan.objective == case["input"]
    assert plan.detected_intents == context.llm_assessment.intents
    assert plan.selected_agents == case["expected_agents"]
    assert [step.agent_id for step in plan.steps] == case["expected_agents"]
    assert plan.data_source_categories == case["expected_data_source_categories"]
    assert plan.approval_surface_id == f"surface_{plan.plan_id}"
    assert plan.immutable_after_approval is True
    assert all(step.expected_output for step in plan.steps)
    assert all(step.data_source_categories for step in plan.steps)
    assert all(step.parallel_group for step in plan.steps[:-1])
    assert plan.steps[-1].agent_id == "synthesis"
    assert plan.steps[-1].depends_on == [step.step_id for step in plan.steps[:-1]]
    assert "Synthetic demo data only" in plan.risk_notes


@pytest.mark.asyncio
async def test_planner_creates_plan_before_specialist_invocation_is_allowed() -> None:
    # Arrange
    registry = AgentRegistry.from_default_config()
    router = RequestRouter(
        slm_client=MockSlmIntentClient(),
        intent_classifier=DeterministicIntentClassifier(),
        registry=registry,
    )
    planner = DraftExecutionPlanner(registry=registry)
    specialist = _FakeSpecialist()

    # Act
    context = await router.route_request(
        "Prepare me for tomorrow's meeting with ABC Manufacturing."
    )
    plan = planner.create_plan(context)
    request = SpecialistRequest(
        request_id="request_before_approval",
        user_input=context.user_input,
        agent_id=plan.steps[0].agent_id,
        plan_id=plan.plan_id,
        step_id=plan.steps[0].step_id,
    )

    # Assert
    assert plan.selected_agents == [
        "relationship_summary",
        "internal_knowledge",
        "industry_research",
        "synthesis",
    ]
    with pytest.raises(
        SpecialistPreApprovalError, match="requires structured approval"
    ):
        await call_specialist_with_guard(context, request, specialist)
    assert specialist.calls == []


def test_planner_excludes_agents_removed_from_current_registry() -> None:
    # Arrange
    available_descriptors = [
        descriptor
        for descriptor in AgentRegistry.from_default_config().descriptors()
        if descriptor.agent_id != "industry_research"
    ]
    registry = _StaticRegistry(available_descriptors)
    context = RequestContext(
        user_input="Prepare me for tomorrow's meeting with ABC Manufacturing.",
        slm_suggestion=IntentSuggestion(intent="meeting_prep", confidence=0.82),
        llm_assessment=LlmIntentAssessment(
            intents=[
                "meeting_prep",
                "relationship_summary",
                "internal_knowledge",
                "industry_research",
            ],
            confidence=0.91,
            complexity="complex",
            required_agents=[
                "relationship_summary",
                "internal_knowledge",
                "industry_research",
                "synthesis",
            ],
            rationale="Meeting preparation requires multiple specialists.",
        ),
        decision=RoutingDecision(
            path="plan_required",
            selected_agent=None,
            confidence=0.874,
            reason="Plan approval required.",
        ),
    )
    planner = DraftExecutionPlanner(registry=registry)

    # Act
    plan = planner.create_plan(context)

    # Assert
    assert "industry_research" not in plan.selected_agents
    assert "industry_research" not in {step.agent_id for step in plan.steps}
    assert plan.selected_agents == [
        "relationship_summary",
        "internal_knowledge",
        "synthesis",
    ]
    assert "Unavailable agents omitted: industry_research." in plan.risk_notes


@pytest.mark.asyncio
async def test_complex_request_requires_approval_plan_route() -> None:
    # Arrange
    user_input = "Prepare me for tomorrow's meeting with ABC Manufacturing."
    slm_suggestion = IntentSuggestion(intent="meeting_prep", confidence=0.82)
    slm_client = _RecordingSlmClient(slm_suggestion)
    classifier = _RecordingClassifier(
        LlmIntentAssessment(
            intents=["meeting_prep", "relationship_summary", "internal_knowledge"],
            confidence=0.91,
            complexity="complex",
            required_agents=[
                "relationship_summary",
                "internal_knowledge",
                "industry_research",
                "synthesis",
            ],
            rationale="Meeting prep requires multiple specialists and synthesis.",
        ),
        slm_client,
    )
    router = RequestRouter(
        slm_client=slm_client,
        intent_classifier=classifier,
        registry=AgentRegistry.from_default_config(),
    )

    # Act
    context = await router.route_request(user_input)

    # Assert
    assert slm_client.calls == [user_input]
    assert classifier.assess_calls == [(user_input, slm_suggestion)]
    assert classifier.slm_call_counts_seen == [1]
    assert context.decision.path == "plan_required"
    assert context.decision.selected_agent is None
    assert context.llm_assessment.required_agents == [
        "relationship_summary",
        "internal_knowledge",
        "industry_research",
        "synthesis",
    ]
    assert "multi" in context.decision.reason.casefold()


@pytest.mark.asyncio
async def test_ambiguous_but_routable_request_returns_plan_required() -> None:
    # Arrange
    user_input = "Help me with ABC."
    router = RequestRouter(
        slm_client=MockSlmIntentClient(),
        intent_classifier=DeterministicIntentClassifier(),
        registry=AgentRegistry.from_default_config(),
    )

    # Act
    context = await router.route_request(user_input)

    # Assert
    assert context.llm_assessment.intents == ["unknown"]
    assert context.llm_assessment.required_agents == ["data_quality"]
    assert context.decision.path == "plan_required"
    assert context.decision.selected_agent is None


@pytest.mark.asyncio
async def test_non_hero_plan_required_route_drafts_plan_before_specialist_call() -> None:
    # Arrange
    user_input = "Help me with ABC."
    registry = AgentRegistry.from_default_config()
    router = RequestRouter(
        slm_client=MockSlmIntentClient(),
        intent_classifier=DeterministicIntentClassifier(),
        registry=registry,
    )
    planner = DraftExecutionPlanner(registry=registry)
    specialist = _FakeSpecialist()

    # Act
    context = await router.route_request(user_input)
    pre_plan_request = SpecialistRequest(
        request_id="request_non_hero_pre_plan",
        user_input=context.user_input,
        agent_id="data_quality",
    )
    with pytest.raises(
        SpecialistPreApprovalError, match="requires structured approval"
    ):
        await call_specialist_with_guard(context, pre_plan_request, specialist)
    plan = planner.create_plan(context)

    # Assert
    assert context.decision.path == "plan_required"
    assert context.llm_assessment.intents == ["unknown"]
    assert context.llm_assessment.required_agents == ["data_quality"]
    assert specialist.calls == []
    assert plan.objective == user_input
    assert plan.detected_intents == ["unknown"]
    assert plan.selected_agents == ["data_quality", "synthesis"]
    assert plan.approval_surface_id == "surface_plan_data_quality"
    assert plan.data_source_categories == ["data_quality"]
    assert "Synthetic demo data only" in plan.risk_notes
    assert [step.agent_id for step in plan.steps] == ["data_quality", "synthesis"]
    assert plan.steps[0].expected_output == "Data quality gaps and clarification needs."
    assert plan.steps[0].parallel_group == "parallel_research"
    assert plan.steps[1].depends_on == [plan.steps[0].step_id]
    assert plan.immutable_after_approval is True


@pytest.mark.asyncio
async def test_clarification_required_only_when_safe_plan_cannot_be_formed() -> None:
    # Arrange
    user_input = "Handle this request with a retired agent."
    slm_suggestion = IntentSuggestion(intent="unknown", confidence=0.4)
    slm_client = _RecordingSlmClient(slm_suggestion)
    classifier = _RecordingClassifier(
        LlmIntentAssessment(
            intents=["unknown"],
            confidence=0.4,
            complexity="complex",
            required_agents=["retired_specialist"],
            rationale="The needed specialist is not available.",
        ),
        slm_client,
    )
    router = RequestRouter(
        slm_client=slm_client,
        intent_classifier=classifier,
        registry=AgentRegistry.from_default_config(),
    )

    # Act
    context = await router.route_request(user_input)

    # Assert
    assert slm_client.calls == [user_input]
    assert classifier.assess_calls == [(user_input, slm_suggestion)]
    assert classifier.slm_call_counts_seen == [1]
    assert context.decision.path == "clarification_required"
    assert context.decision.selected_agent is None
    assert "unavailable" in context.decision.reason.casefold()


@pytest.mark.asyncio
async def test_specialist_guard_raises_before_complex_plan_approval() -> None:
    # Arrange
    router = RequestRouter(
        slm_client=MockSlmIntentClient(),
        intent_classifier=DeterministicIntentClassifier(),
        registry=AgentRegistry.from_default_config(),
    )
    context = await router.route_request(
        "Prepare me for tomorrow's meeting with ABC Manufacturing."
    )
    specialist = _FakeSpecialist()
    request = SpecialistRequest(
        request_id="request_preapproval",
        user_input=context.user_input,
        agent_id="internal_knowledge",
        plan_id="plan_meeting_prep",
        step_id="step_internal_knowledge",
    )

    # Act / Assert
    with pytest.raises(
        SpecialistPreApprovalError, match="requires structured approval"
    ):
        await call_specialist_with_guard(context, request, specialist)

    assert specialist.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("plan_id", [None, "plan_other"])
async def test_specialist_guard_requires_request_plan_id_to_match_approved_plan(
    plan_id: str | None,
) -> None:
    # Arrange
    router = RequestRouter(
        slm_client=MockSlmIntentClient(),
        intent_classifier=DeterministicIntentClassifier(),
        registry=AgentRegistry.from_default_config(),
    )
    context = await router.route_request(
        "Prepare me for tomorrow's meeting with ABC Manufacturing."
    )
    context.mark_plan_approved("plan_meeting_prep")
    specialist = _FakeSpecialist()
    request = SpecialistRequest(
        request_id="request_plan_binding",
        user_input=context.user_input,
        agent_id="internal_knowledge",
        plan_id=plan_id,
        step_id="step_internal_knowledge",
    )

    # Act / Assert
    with pytest.raises(SpecialistPreApprovalError, match="approved plan_id"):
        await call_specialist_with_guard(context, request, specialist)

    assert specialist.calls == []


@pytest.mark.asyncio
async def test_specialist_guard_allows_approved_plan_matching_request() -> None:
    # Arrange
    router = RequestRouter(
        slm_client=MockSlmIntentClient(),
        intent_classifier=DeterministicIntentClassifier(),
        registry=AgentRegistry.from_default_config(),
    )
    context = await router.route_request(
        "Prepare me for tomorrow's meeting with ABC Manufacturing."
    )
    context.mark_plan_approved("plan_meeting_prep")
    specialist = _FakeSpecialist()
    request = SpecialistRequest(
        request_id="request_approved_plan",
        user_input=context.user_input,
        agent_id="internal_knowledge",
        plan_id="plan_meeting_prep",
        step_id="step_internal_knowledge",
    )

    # Act
    response = await call_specialist_with_guard(context, request, specialist)

    # Assert
    assert response.agent_id == "internal_knowledge"
    assert specialist.calls == [request]


@pytest.mark.asyncio
async def test_specialist_guard_allows_direct_route_specialist_call() -> None:
    # Arrange
    router = RequestRouter(
        slm_client=MockSlmIntentClient(),
        intent_classifier=DeterministicIntentClassifier(),
        registry=AgentRegistry.from_default_config(),
    )
    context = await router.route_request(
        "Summarize the internal notes for ABC Manufacturing."
    )
    specialist = _FakeSpecialist()
    request = SpecialistRequest(
        request_id="request_direct",
        user_input=context.user_input,
        agent_id="internal_knowledge",
    )

    # Act
    response = await call_specialist_with_guard(context, request, specialist)

    # Assert
    assert response.agent_id == "internal_knowledge"
    assert specialist.calls == [request]
