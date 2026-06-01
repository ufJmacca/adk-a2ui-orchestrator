import json
from pathlib import Path
from typing import Any

import pytest

from orchestrator_demo.contracts import (
    AgentDescriptor,
    ExecutionPlan,
    IntentSuggestion,
    LlmIntentAssessment,
    PlanStep,
    RoutingDecision,
    SpecialistRequest,
    SpecialistResponse,
)
from orchestrator_demo.intent.classifier import DeterministicIntentClassifier
from orchestrator_demo.intent.slm_mock_client import MockSlmIntentClient
from orchestrator_demo.orchestrator.request_context import (
    PlanApprovalStateError,
    RequestContext,
    SpecialistPreApprovalError,
    call_specialist_with_guard,
)
from orchestrator_demo.orchestrator.planner import DraftExecutionPlanner, PlanRequiredError
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


def _step_for(plan: ExecutionPlan, agent_id: str) -> PlanStep:
    for step in plan.steps:
        if step.agent_id == agent_id:
            return step

    raise AssertionError(f"plan does not include agent {agent_id!r}")


def _approved_context_for_step(plan: ExecutionPlan, step: PlanStep) -> dict[str, Any]:
    context: dict[str, Any] = {
        "objective": plan.objective,
        "planVersion": plan.plan_version,
        "expectedOutput": step.expected_output,
        "dataSourceCategories": list(step.data_source_categories),
        "dependsOn": list(step.depends_on),
    }
    if step.parallel_group is not None:
        context["parallelGroup"] = step.parallel_group

    return context


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
    assert plan.plan_id.startswith(f"plan_{context.llm_assessment.intents[0]}_")
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


def test_planner_fails_when_required_synthesis_is_unavailable() -> None:
    # Arrange
    available_descriptors = [
        descriptor
        for descriptor in AgentRegistry.from_default_config().descriptors()
        if descriptor.agent_id != "synthesis"
    ]
    registry = _StaticRegistry(available_descriptors)
    context = RequestContext(
        user_input="Help me with ABC.",
        slm_suggestion=IntentSuggestion(intent="unknown", confidence=0.4),
        llm_assessment=LlmIntentAssessment(
            intents=["unknown"],
            confidence=0.4,
            complexity="complex",
            required_agents=["data_quality"],
            rationale="Ambiguous request requires data quality and synthesis.",
        ),
        decision=RoutingDecision(
            path="plan_required",
            selected_agent=None,
            confidence=0.4,
            reason="Plan approval required.",
        ),
    )
    planner = DraftExecutionPlanner(registry=registry)

    # Act / Assert
    with pytest.raises(PlanRequiredError, match="unavailable synthesis"):
        planner.create_plan(context)


def test_planner_fails_cleanly_when_all_selected_agents_are_unavailable() -> None:
    # Arrange
    available_descriptors = [
        descriptor
        for descriptor in AgentRegistry.from_default_config().descriptors()
        if descriptor.agent_id != "internal_knowledge"
    ]
    registry = _StaticRegistry(available_descriptors)
    context = RequestContext(
        user_input="Summarize the internal notes for ABC Manufacturing.",
        slm_suggestion=IntentSuggestion(intent="internal_knowledge", confidence=0.82),
        llm_assessment=LlmIntentAssessment(
            intents=["internal_knowledge"],
            confidence=0.86,
            complexity="simple",
            required_agents=["internal_knowledge"],
            rationale="One specialist can handle the request.",
        ),
        decision=RoutingDecision(
            path="plan_required",
            selected_agent=None,
            confidence=0.844,
            reason="Plan approval required: below direct-route confidence threshold.",
        ),
    )
    planner = DraftExecutionPlanner(registry=registry)

    # Act / Assert
    with pytest.raises(PlanRequiredError, match="requested agents are unavailable"):
        planner.create_plan(context)


def test_planner_fails_when_only_synthesis_remains_available() -> None:
    # Arrange
    registry = _StaticRegistry([_descriptor("synthesis")])
    context = RequestContext(
        user_input="Prepare me for tomorrow's meeting with ABC Manufacturing.",
        slm_suggestion=IntentSuggestion(intent="meeting_prep", confidence=0.82),
        llm_assessment=LlmIntentAssessment(
            intents=[
                "meeting_prep",
                "internal_knowledge",
            ],
            confidence=0.91,
            complexity="complex",
            required_agents=[
                "internal_knowledge",
                "synthesis",
            ],
            rationale="Meeting preparation requires specialist input and synthesis.",
        ),
        decision=RoutingDecision(
            path="plan_required",
            selected_agent=None,
            confidence=0.874,
            reason="Plan approval required.",
        ),
    )
    planner = DraftExecutionPlanner(registry=registry)

    # Act / Assert
    with pytest.raises(PlanRequiredError, match="non-synthesis workstream agent"):
        planner.create_plan(context)


def test_planner_moves_classifier_synthesis_selection_after_workstreams() -> None:
    # Arrange
    registry = _StaticRegistry([_descriptor("synthesis"), _descriptor("web_search")])
    context = RequestContext(
        user_input="Research public information and synthesize the findings.",
        slm_suggestion=IntentSuggestion(intent="web_search", confidence=0.8),
        llm_assessment=LlmIntentAssessment(
            intents=["web_search"],
            confidence=0.9,
            complexity="complex",
            required_agents=["synthesis", "web_search"],
            rationale="A model-backed classifier returned synthesis first.",
        ),
        decision=RoutingDecision(
            path="plan_required",
            selected_agent=None,
            confidence=0.86,
            reason="Plan approval required.",
        ),
    )

    # Act
    plan = DraftExecutionPlanner(registry=registry).create_plan(context)

    # Assert
    assert plan.selected_agents == ["web_search", "synthesis"]
    assert [step.agent_id for step in plan.steps] == ["web_search", "synthesis"]
    assert plan.steps[-1].depends_on == ["step_web_search"]


def test_planner_generates_unique_step_ids_after_slugging_agent_ids() -> None:
    # Arrange
    registry = _StaticRegistry(
        [
            _descriptor("foo bar"),
            _descriptor("foo_bar"),
            _descriptor("synthesis"),
        ]
    )
    context = RequestContext(
        user_input="Compare two similarly named workstreams.",
        slm_suggestion=IntentSuggestion(intent="unknown", confidence=0.6),
        llm_assessment=LlmIntentAssessment(
            intents=["unknown"],
            confidence=0.8,
            complexity="complex",
            required_agents=["foo bar", "foo_bar", "synthesis"],
            rationale="Two available specialists and synthesis are needed.",
        ),
        decision=RoutingDecision(
            path="plan_required",
            selected_agent=None,
            confidence=0.72,
            reason="Plan approval required.",
        ),
    )

    # Act
    plan = DraftExecutionPlanner(registry=registry).create_plan(context)

    # Assert
    step_ids = [step.step_id for step in plan.steps]
    assert step_ids == ["step_foo_bar", "step_foo_bar_2", "step_synthesis"]
    assert len(step_ids) == len(set(step_ids))
    assert plan.steps[-1].depends_on == ["step_foo_bar", "step_foo_bar_2"]


@pytest.mark.asyncio
async def test_planner_generates_request_scoped_plan_and_surface_ids() -> None:
    # Arrange
    user_input = "Prepare me for tomorrow's meeting with ABC Manufacturing."
    registry = AgentRegistry.from_default_config()
    router = RequestRouter(
        slm_client=MockSlmIntentClient(),
        intent_classifier=DeterministicIntentClassifier(),
        registry=registry,
    )
    planner = DraftExecutionPlanner(registry=registry)

    # Act
    first_plan = planner.create_plan(await router.route_request(user_input))
    second_plan = planner.create_plan(await router.route_request(user_input))

    # Assert
    assert first_plan.plan_id.startswith("plan_meeting_prep_")
    assert second_plan.plan_id.startswith("plan_meeting_prep_")
    assert first_plan.plan_id != second_plan.plan_id
    assert first_plan.approval_surface_id == f"surface_{first_plan.plan_id}"
    assert second_plan.approval_surface_id == f"surface_{second_plan.plan_id}"
    assert first_plan.approval_surface_id != second_plan.approval_surface_id


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
    assert plan.plan_id.startswith("plan_data_quality_")
    assert plan.approval_surface_id == f"surface_{plan.plan_id}"
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
    planner = DraftExecutionPlanner(registry=AgentRegistry.from_default_config())
    approved_plan = planner.create_plan(context)
    approved_step = _step_for(approved_plan, "internal_knowledge")
    context.mark_plan_approved(approved_plan)
    specialist = _FakeSpecialist()
    request = SpecialistRequest(
        request_id="request_plan_binding",
        user_input=context.user_input,
        agent_id=approved_step.agent_id,
        plan_id=plan_id,
        step_id=approved_step.step_id,
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
    planner = DraftExecutionPlanner(registry=AgentRegistry.from_default_config())
    approved_plan = planner.create_plan(context)
    approved_step = _step_for(approved_plan, "internal_knowledge")
    context.mark_plan_approved(approved_plan)
    specialist = _FakeSpecialist()
    request = SpecialistRequest(
        request_id="request_approved_plan",
        user_input=approved_step.instruction,
        agent_id=approved_step.agent_id,
        plan_id=approved_plan.plan_id,
        step_id=approved_step.step_id,
        context=_approved_context_for_step(approved_plan, approved_step),
    )

    # Act
    response = await call_specialist_with_guard(context, request, specialist)

    # Assert
    assert response.agent_id == "internal_knowledge"
    assert specialist.calls == [request]


@pytest.mark.asyncio
async def test_mark_plan_approved_rejects_foreign_plan_before_first_approval() -> None:
    # Arrange
    user_input = "Prepare me for tomorrow's meeting with ABC Manufacturing."
    registry = AgentRegistry.from_default_config()
    router = RequestRouter(
        slm_client=MockSlmIntentClient(),
        intent_classifier=DeterministicIntentClassifier(),
        registry=registry,
    )
    planner = DraftExecutionPlanner(registry=registry)
    context = await router.route_request(user_input)
    foreign_context = await router.route_request(user_input)
    draft_plan = planner.create_plan(context)
    foreign_plan = planner.create_plan(foreign_context)

    # Act / Assert
    with pytest.raises(PlanApprovalStateError, match="plan_scope_id"):
        context.mark_plan_approved(foreign_plan)

    assert context.approved_plan_id is None
    context.mark_plan_approved(draft_plan)
    assert context.approved_plan_id == draft_plan.plan_id


@pytest.mark.asyncio
async def test_mark_plan_approved_rejects_plan_that_differs_from_current_draft() -> None:
    # Arrange
    registry = AgentRegistry.from_default_config()
    router = RequestRouter(
        slm_client=MockSlmIntentClient(),
        intent_classifier=DeterministicIntentClassifier(),
        registry=registry,
    )
    context = await router.route_request(
        "Prepare me for tomorrow's meeting with ABC Manufacturing."
    )
    draft_plan = DraftExecutionPlanner(registry=registry).create_plan(context)
    draft_step = _step_for(draft_plan, "internal_knowledge")
    tampered_steps = [
        step.model_copy(
            update={"instruction": "Research an unapproved different customer."}
        )
        if step.step_id == draft_step.step_id
        else step
        for step in draft_plan.steps
    ]
    tampered_plan = ExecutionPlan(
        **{
            **draft_plan.model_dump(),
            "steps": [step.model_dump() for step in tampered_steps],
        }
    )

    # Act / Assert
    with pytest.raises(PlanApprovalStateError, match="current draft"):
        context.mark_plan_approved(tampered_plan)

    assert context.approved_plan_id is None


@pytest.mark.asyncio
async def test_mark_plan_approved_does_not_overwrite_frozen_plan_payloads() -> None:
    # Arrange
    registry = AgentRegistry.from_default_config()
    router = RequestRouter(
        slm_client=MockSlmIntentClient(),
        intent_classifier=DeterministicIntentClassifier(),
        registry=registry,
    )
    context = await router.route_request(
        "Prepare me for tomorrow's meeting with ABC Manufacturing."
    )
    approved_plan = DraftExecutionPlanner(registry=registry).create_plan(context)
    approved_step = _step_for(approved_plan, "internal_knowledge")
    context.mark_plan_approved(approved_plan)
    tampered_instruction = "Research a different customer that was not approved."
    tampered_steps = [
        step.model_copy(update={"instruction": tampered_instruction})
        if step.step_id == approved_step.step_id
        else step
        for step in approved_plan.steps
    ]
    tampered_plan = ExecutionPlan(
        **{
            **approved_plan.model_dump(),
            "steps": [step.model_dump() for step in tampered_steps],
        }
    )
    tampered_step = _step_for(tampered_plan, "internal_knowledge")
    specialist = _FakeSpecialist()
    tampered_request = SpecialistRequest(
        request_id="request_tampered_approved_plan",
        user_input=tampered_step.instruction,
        agent_id=tampered_step.agent_id,
        plan_id=tampered_plan.plan_id,
        step_id=tampered_step.step_id,
        context=_approved_context_for_step(tampered_plan, tampered_step),
    )

    # Act / Assert
    with pytest.raises(PlanApprovalStateError, match="immutable"):
        context.mark_plan_approved(tampered_plan)
    with pytest.raises(SpecialistPreApprovalError, match="approved instruction"):
        await call_specialist_with_guard(context, tampered_request, specialist)

    assert specialist.calls == []


@pytest.mark.asyncio
async def test_specialist_guard_rejects_changed_user_input_for_approved_step() -> None:
    # Arrange
    registry = AgentRegistry.from_default_config()
    router = RequestRouter(
        slm_client=MockSlmIntentClient(),
        intent_classifier=DeterministicIntentClassifier(),
        registry=registry,
    )
    context = await router.route_request(
        "Prepare me for tomorrow's meeting with ABC Manufacturing."
    )
    approved_plan = DraftExecutionPlanner(registry=registry).create_plan(context)
    approved_step = _step_for(approved_plan, "internal_knowledge")
    context.mark_plan_approved(approved_plan)
    specialist = _FakeSpecialist()
    request = SpecialistRequest(
        request_id="request_changed_approved_instruction",
        user_input="Research a different customer that was not approved.",
        agent_id=approved_step.agent_id,
        plan_id=approved_plan.plan_id,
        step_id=approved_step.step_id,
        context=_approved_context_for_step(approved_plan, approved_step),
    )

    # Act / Assert
    with pytest.raises(SpecialistPreApprovalError, match="approved instruction"):
        await call_specialist_with_guard(context, request, specialist)

    assert specialist.calls == []


@pytest.mark.asyncio
async def test_specialist_guard_rejects_changed_context_for_approved_step() -> None:
    # Arrange
    registry = AgentRegistry.from_default_config()
    router = RequestRouter(
        slm_client=MockSlmIntentClient(),
        intent_classifier=DeterministicIntentClassifier(),
        registry=registry,
    )
    context = await router.route_request(
        "Prepare me for tomorrow's meeting with ABC Manufacturing."
    )
    approved_plan = DraftExecutionPlanner(registry=registry).create_plan(context)
    approved_step = _step_for(approved_plan, "internal_knowledge")
    context.mark_plan_approved(approved_plan)
    specialist = _FakeSpecialist()
    changed_context = _approved_context_for_step(approved_plan, approved_step)
    changed_context["objective"] = "Prepare for an unapproved different customer."
    request = SpecialistRequest(
        request_id="request_changed_approved_context",
        user_input=approved_step.instruction,
        agent_id=approved_step.agent_id,
        plan_id=approved_plan.plan_id,
        step_id=approved_step.step_id,
        context=changed_context,
    )

    # Act / Assert
    with pytest.raises(SpecialistPreApprovalError, match="approved context"):
        await call_specialist_with_guard(context, request, specialist)

    assert specialist.calls == []


@pytest.mark.asyncio
async def test_specialist_guard_rejects_unapproved_step_for_approved_plan() -> None:
    # Arrange
    registry = AgentRegistry.from_default_config()
    router = RequestRouter(
        slm_client=MockSlmIntentClient(),
        intent_classifier=DeterministicIntentClassifier(),
        registry=registry,
    )
    context = await router.route_request(
        "Prepare me for tomorrow's meeting with ABC Manufacturing."
    )
    approved_plan = DraftExecutionPlanner(registry=registry).create_plan(context)
    context.mark_plan_approved(approved_plan)
    specialist = _FakeSpecialist()
    request = SpecialistRequest(
        request_id="request_unapproved_step",
        user_input=context.user_input,
        agent_id="credit_risk",
        plan_id=approved_plan.plan_id,
        step_id="step_credit_risk",
    )

    # Act / Assert
    with pytest.raises(SpecialistPreApprovalError, match="approved step_id"):
        await call_specialist_with_guard(context, request, specialist)

    assert specialist.calls == []


@pytest.mark.asyncio
async def test_specialist_guard_rejects_agent_mismatch_for_approved_step() -> None:
    # Arrange
    registry = AgentRegistry.from_default_config()
    router = RequestRouter(
        slm_client=MockSlmIntentClient(),
        intent_classifier=DeterministicIntentClassifier(),
        registry=registry,
    )
    context = await router.route_request(
        "Prepare me for tomorrow's meeting with ABC Manufacturing."
    )
    approved_plan = DraftExecutionPlanner(registry=registry).create_plan(context)
    approved_step = _step_for(approved_plan, "internal_knowledge")
    context.mark_plan_approved(approved_plan)
    specialist = _FakeSpecialist()
    request = SpecialistRequest(
        request_id="request_unapproved_agent",
        user_input=context.user_input,
        agent_id="credit_risk",
        plan_id=approved_plan.plan_id,
        step_id=approved_step.step_id,
    )

    # Act / Assert
    with pytest.raises(SpecialistPreApprovalError, match="agent_id must match"):
        await call_specialist_with_guard(context, request, specialist)

    assert specialist.calls == []


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("plan_id", "step_id"),
    [
        ("plan_unapproved", None),
        (None, "step_unapproved"),
        ("plan_unapproved", "step_unapproved"),
    ],
)
async def test_specialist_guard_rejects_direct_route_plan_identifiers(
    plan_id: str | None,
    step_id: str | None,
) -> None:
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
        request_id="request_direct_with_plan_scope",
        user_input=context.user_input,
        agent_id="internal_knowledge",
        plan_id=plan_id,
        step_id=step_id,
    )

    # Act / Assert
    with pytest.raises(SpecialistPreApprovalError, match="must not include plan_id"):
        await call_specialist_with_guard(context, request, specialist)

    assert specialist.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("changed_user_input", "changed_context", "error_match"),
    [
        (
            "Summarize the internal notes for a different customer.",
            {},
            "original routed user_input",
        ),
        (
            "Summarize the internal notes for ABC Manufacturing.",
            {"customer": "Different Customer"},
            "original routed context",
        ),
    ],
)
async def test_specialist_guard_rejects_direct_route_payload_reuse(
    changed_user_input: str,
    changed_context: dict[str, Any],
    error_match: str,
) -> None:
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
        request_id="request_direct_reused",
        user_input=changed_user_input,
        agent_id="internal_knowledge",
        context=changed_context,
    )

    # Act / Assert
    with pytest.raises(SpecialistPreApprovalError, match=error_match):
        await call_specialist_with_guard(context, request, specialist)

    assert specialist.calls == []
