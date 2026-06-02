import json
from pathlib import Path
from typing import Any

import pytest

from orchestrator_demo.agents import build_default_specialists
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
from orchestrator_demo.a2ui_support.event_parser import parse_user_action
from orchestrator_demo.orchestrator.planner import (
    DraftExecutionPlanner,
    PlanCreationError,
)
from orchestrator_demo.orchestrator.approval_state import ApprovalStateStore
from orchestrator_demo.orchestrator.request_context import (
    PlanApprovalStateError,
    RequestContext,
    SpecialistPreApprovalError,
    call_specialist_with_guard,
)
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


def _write_registry_config(path: Path, agent_ids: list[str]) -> None:
    descriptor_sources = [
        f"""AgentDescriptor(
        agent_id={agent_id!r},
        display_name={agent_id.replace("_", " ").title()!r},
        capabilities=["business banking support"],
        input_schema={{"type": "object"}},
        output_schema={{"type": "object"}},
        a2ui_catalogs=["basic"],
        routing_examples=["Handle a {agent_id} request."],
        execution_mode="local_llm",
    )"""
        for agent_id in agent_ids
    ]
    path.write_text(
        "from orchestrator_demo.contracts import AgentDescriptor\n\n"
        "AVAILABLE_AGENTS = [\n"
        + ",\n".join(descriptor_sources)
        + "\n]\n",
        encoding="utf-8",
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


@pytest.mark.asyncio
async def test_router_allows_partial_complex_plan_after_registry_reload(
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
    _write_registry_config(
        config_path,
        ["relationship_summary", "internal_knowledge", "synthesis"],
    )
    registry.reload()
    router = RequestRouter(
        slm_client=MockSlmIntentClient(),
        intent_classifier=DeterministicIntentClassifier(),
        registry=registry,
    )
    planner = DraftExecutionPlanner(registry=registry)

    # Act
    context = await router.route_request(
        "Prepare me for tomorrow's meeting with ABC Manufacturing."
    )
    plan = planner.create_plan(context)

    # Assert
    assert context.decision.path == "plan_required"
    assert "industry_research" in context.llm_assessment.required_agents
    assert "industry_research" not in plan.selected_agents
    assert "industry_research" not in {step.agent_id for step in plan.steps}
    assert plan.selected_agents == [
        "relationship_summary",
        "internal_knowledge",
        "synthesis",
    ]
    assert "Unavailable agents omitted: industry_research." in plan.risk_notes


@pytest.mark.parametrize(
    ("available_agent_ids", "required_agent_ids", "complexity"),
    [
        pytest.param([], ["relationship_summary"], "simple", id="empty"),
        pytest.param(
            ["synthesis"],
            ["relationship_summary", "synthesis"],
            "complex",
            id="synthesis-only",
        ),
    ],
)
def test_planner_fails_when_filtering_leaves_no_executable_workstream(
    available_agent_ids: list[str],
    required_agent_ids: list[str],
    complexity: str,
) -> None:
    # Arrange
    context = RequestContext(
        user_input="Prepare context using an agent that was just removed.",
        slm_suggestion=IntentSuggestion(intent="relationship_summary", confidence=0.7),
        llm_assessment=LlmIntentAssessment(
            intents=["relationship_summary"],
            confidence=0.8,
            complexity=complexity,
            required_agents=required_agent_ids,
            rationale="A specialist workstream was required before registry reload.",
        ),
        decision=RoutingDecision(
            path="plan_required",
            selected_agent=None,
            confidence=0.76,
            reason="Plan approval required.",
        ),
    )
    planner = DraftExecutionPlanner(
        registry=_StaticRegistry(
            [_descriptor(agent_id) for agent_id in available_agent_ids]
        )
    )

    # Act / Assert
    with pytest.raises(
        PlanCreationError,
        match="no available non-synthesis specialist workstream",
    ):
        planner.create_plan(context)


def test_planner_fails_when_implicit_synthesis_agent_is_unavailable() -> None:
    # Arrange
    available_descriptors = [
        descriptor
        for descriptor in AgentRegistry.from_default_config().descriptors()
        if descriptor.agent_id != "synthesis"
    ]
    context = RequestContext(
        user_input="Help me with ABC.",
        slm_suggestion=IntentSuggestion(intent="unknown", confidence=0.4),
        llm_assessment=LlmIntentAssessment(
            intents=["unknown"],
            confidence=0.44,
            complexity="complex",
            required_agents=["data_quality"],
            rationale="Ambiguous requests require data quality review.",
        ),
        decision=RoutingDecision(
            path="plan_required",
            selected_agent=None,
            confidence=0.424,
            reason="Plan approval required.",
        ),
    )
    planner = DraftExecutionPlanner(registry=_StaticRegistry(available_descriptors))

    # Act / Assert
    with pytest.raises(
        PlanCreationError,
        match="requires synthesis but the synthesis agent is unavailable",
    ):
        planner.create_plan(context)


@pytest.mark.asyncio
async def test_router_requires_implicit_synthesis_before_plan_required() -> None:
    # Arrange
    available_descriptors = [
        descriptor
        for descriptor in AgentRegistry.from_default_config().descriptors()
        if descriptor.agent_id != "synthesis"
    ]
    user_input = "Help me understand what information is missing for ABC."
    slm_suggestion = IntentSuggestion(intent="unknown", confidence=0.4)
    slm_client = _RecordingSlmClient(slm_suggestion)
    classifier = _RecordingClassifier(
        LlmIntentAssessment(
            intents=["unknown"],
            confidence=0.44,
            complexity="complex",
            required_agents=["data_quality"],
            rationale="Ambiguous requests require a complex data quality review.",
        ),
        slm_client,
    )
    router = RequestRouter(
        slm_client=slm_client,
        intent_classifier=classifier,
        registry=_StaticRegistry(available_descriptors),
    )

    # Act
    context = await router.route_request(user_input)

    # Assert
    assert slm_client.calls == [user_input]
    assert classifier.assess_calls == [(user_input, slm_suggestion)]
    assert context.llm_assessment.required_agents == ["data_quality"]
    assert context.decision.path == "clarification_required"
    assert context.decision.selected_agent is None
    assert "synthesis" in context.decision.reason
    assert "unavailable" in context.decision.reason.casefold()


def test_planner_does_not_require_synthesis_for_duplicate_single_workstream() -> None:
    # Arrange
    context = RequestContext(
        user_input="Check whether ABC Manufacturing needs better data.",
        slm_suggestion=IntentSuggestion(intent="data_quality", confidence=0.72),
        llm_assessment=LlmIntentAssessment(
            intents=["data_quality"],
            confidence=0.74,
            complexity="simple",
            required_agents=["data_quality", "data_quality"],
            rationale="Low confidence requires approval, but only one workstream.",
        ),
        decision=RoutingDecision(
            path="plan_required",
            selected_agent=None,
            confidence=0.732,
            reason="Plan approval required.",
        ),
    )
    planner = DraftExecutionPlanner(
        registry=_StaticRegistry([_descriptor("data_quality")])
    )

    # Act
    plan = planner.create_plan(context)

    # Assert
    assert plan.selected_agents == ["data_quality"]
    assert [step.agent_id for step in plan.steps] == ["data_quality"]
    assert plan.steps[0].depends_on == []


def test_planner_moves_selected_synthesis_after_specialist_steps() -> None:
    # Arrange
    context = RequestContext(
        user_input="Review credit risk and summarize the result for ABC Manufacturing.",
        slm_suggestion=IntentSuggestion(intent="credit_risk", confidence=0.82),
        llm_assessment=LlmIntentAssessment(
            intents=["credit_risk"],
            confidence=0.91,
            complexity="complex",
            required_agents=["synthesis", "credit_risk"],
            rationale="Credit risk review requires final synthesis.",
        ),
        decision=RoutingDecision(
            path="plan_required",
            selected_agent=None,
            confidence=0.874,
            reason="Plan approval required.",
        ),
    )
    planner = DraftExecutionPlanner(registry=AgentRegistry.from_default_config())

    # Act
    plan = planner.create_plan(context)

    # Assert
    assert plan.selected_agents == ["credit_risk", "synthesis"]
    assert [step.agent_id for step in plan.steps] == ["credit_risk", "synthesis"]
    assert plan.steps[0].depends_on == []
    assert plan.steps[1].depends_on == [plan.steps[0].step_id]


def test_planner_generates_unique_step_ids_for_slug_colliding_agent_ids() -> None:
    # Arrange
    context = RequestContext(
        user_input="Coordinate the custom agent review for ABC Manufacturing.",
        slm_suggestion=IntentSuggestion(intent="unknown", confidence=0.6),
        llm_assessment=LlmIntentAssessment(
            intents=["unknown"],
            confidence=0.9,
            complexity="complex",
            required_agents=["foo bar", "foo_bar", "synthesis"],
            rationale="Dynamic agents need combined review and synthesis.",
        ),
        decision=RoutingDecision(
            path="plan_required",
            selected_agent=None,
            confidence=0.78,
            reason="Plan approval required.",
        ),
    )
    planner = DraftExecutionPlanner(
        registry=_StaticRegistry(
            [_descriptor("foo bar"), _descriptor("foo_bar"), _descriptor("synthesis")]
        )
    )

    # Act
    plan = planner.create_plan(context)

    # Assert
    assert plan.selected_agents == ["foo bar", "foo_bar", "synthesis"]
    assert [step.agent_id for step in plan.steps] == [
        "foo bar",
        "foo_bar",
        "synthesis",
    ]
    assert [step.step_id for step in plan.steps] == [
        "step_foo_bar",
        "step_foo_bar_2",
        "step_synthesis",
    ]
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
@pytest.mark.asyncio
@pytest.mark.parametrize("runtime_key", ["dependencyOutputs", "stepResults"])
async def test_specialist_guard_allows_dependency_outputs_for_approved_step(
    runtime_key: str,
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
    approved_step = approved_plan.steps[-1]
    dependency_outputs = {
        dependency_id: {"content": f"completed output for {dependency_id}"}
        for dependency_id in approved_step.depends_on
    }
    runtime_context = _approved_context_for_step(approved_plan, approved_step)
    runtime_context[runtime_key] = dependency_outputs
    context.mark_plan_approved(approved_plan)
    specialist = _FakeSpecialist()
    request = SpecialistRequest(
        request_id="request_approved_plan_dependency_outputs",
        user_input=approved_step.instruction,
        agent_id=approved_step.agent_id,
        plan_id=approved_plan.plan_id,
        step_id=approved_step.step_id,
        context=runtime_context,
    )

    # Act
    response = await call_specialist_with_guard(context, request, specialist)

    # Assert
    assert response.agent_id == "synthesis"
    assert specialist.calls == [request]
    assert specialist.calls[0].context[runtime_key] == dependency_outputs


@pytest.mark.asyncio
async def test_specialist_guard_rejects_missing_dependency_outputs_for_dependent_step() -> None:
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
    approved_step = approved_plan.steps[-1]
    runtime_context = _approved_context_for_step(approved_plan, approved_step)
    context.mark_plan_approved(approved_plan)
    specialist = _FakeSpecialist()
    request = SpecialistRequest(
        request_id="request_missing_dependency_outputs",
        user_input=approved_step.instruction,
        agent_id=approved_step.agent_id,
        plan_id=approved_plan.plan_id,
        step_id=approved_step.step_id,
        context=runtime_context,
    )

    # Act / Assert
    with pytest.raises(SpecialistPreApprovalError, match="approved context"):
        await call_specialist_with_guard(context, request, specialist)

    assert specialist.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("runtime_key", ["dependencyOutputs", "stepResults"])
async def test_specialist_guard_rejects_partial_dependency_outputs_for_dependent_step(
    runtime_key: str,
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
    approved_step = approved_plan.steps[-1]
    partial_dependency_outputs = {
        approved_step.depends_on[0]: {
            "content": f"completed output for {approved_step.depends_on[0]}"
        }
    }
    runtime_context = _approved_context_for_step(approved_plan, approved_step)
    runtime_context[runtime_key] = partial_dependency_outputs
    context.mark_plan_approved(approved_plan)
    specialist = _FakeSpecialist()
    request = SpecialistRequest(
        request_id="request_partial_dependency_outputs",
        user_input=approved_step.instruction,
        agent_id=approved_step.agent_id,
        plan_id=approved_plan.plan_id,
        step_id=approved_step.step_id,
        context=runtime_context,
    )

    # Act / Assert
    with pytest.raises(SpecialistPreApprovalError, match="approved context"):
        await call_specialist_with_guard(context, request, specialist)

    assert specialist.calls == []


@pytest.mark.asyncio
async def test_edited_draft_result_syncs_context_before_approval_execution() -> None:
    # Arrange
    from orchestrator_demo.orchestrator.approval_state import ApprovalStateStore

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
    store = ApprovalStateStore(agent_descriptors=registry.descriptors())
    store.add_draft(draft_plan)
    edit_event = {
        "userAction": {
            "type": "add_instruction",
            "surfaceId": draft_plan.approval_surface_id,
            "payload": {
                "planId": draft_plan.plan_id,
                "editedPlanVersion": draft_plan.plan_version,
                "stepId": "step_internal_knowledge",
                "instruction": "Prioritize covenant follow-ups.",
            },
        }
    }

    # Act
    edit_result = store.apply_user_action(edit_event)
    assert edit_result.draft_plan is not None
    context.record_draft_plan(edit_result.draft_plan)
    approve_event = {
        "userAction": {
            "type": "approve_plan",
            "surfaceId": edit_result.draft_plan.approval_surface_id,
            "payload": {
                "planId": edit_result.draft_plan.plan_id,
                "editedPlanVersion": edit_result.draft_plan.plan_version,
                "approvedStepIds": [
                    step.step_id for step in edit_result.draft_plan.steps
                ],
            },
        }
    }
    approval_result = store.apply_user_action(approve_event)
    assert approval_result.approved_plan is not None
    context.mark_plan_approved(approval_result.approved_plan)
    approved_step = _step_for(approval_result.approved_plan, "internal_knowledge")
    specialist = _FakeSpecialist()
    request = SpecialistRequest(
        request_id="request_approved_edited_plan",
        user_input=approved_step.instruction,
        agent_id=approved_step.agent_id,
        plan_id=approval_result.approved_plan.plan_id,
        step_id=approved_step.step_id,
        context=_approved_context_for_step(
            approval_result.approved_plan,
            approved_step,
        ),
    )
    response = await call_specialist_with_guard(context, request, specialist)

    # Assert
    assert edit_result.status == "draft_updated"
    assert edit_result.draft_plan.plan_version == draft_plan.plan_version + 1
    assert approval_result.status == "approved"
    assert context.approved_plan_id == approval_result.approved_plan.plan_id
    assert "Additional instruction: Prioritize covenant follow-ups." in (
        approved_step.instruction
    )
    assert response.agent_id == "internal_knowledge"
    assert specialist.calls == [request]


@pytest.mark.asyncio
async def test_approved_step_payload_context_is_immutable_after_approval() -> None:
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
    approved_payloads = context.approved_plan_step_payloads
    approved_step_payloads = approved_payloads[approved_plan.plan_id]
    stored_context = approved_step_payloads[approved_step.step_id].context
    specialist = _FakeSpecialist()
    tampered_context = _approved_context_for_step(approved_plan, approved_step)
    tampered_context["objective"] = "Prepare for an unapproved different customer."
    tampered_request = SpecialistRequest(
        request_id="request_mutated_approved_context",
        user_input=approved_step.instruction,
        agent_id=approved_step.agent_id,
        plan_id=approved_plan.plan_id,
        step_id=approved_step.step_id,
        context=tampered_context,
    )

    # Act / Assert
    with pytest.raises(TypeError):
        approved_payloads[approved_plan.plan_id] = approved_step_payloads
    with pytest.raises(TypeError):
        approved_step_payloads[approved_step.step_id] = approved_step_payloads[
            approved_step.step_id
        ]
    with pytest.raises(TypeError):
        stored_context["objective"] = "Prepare for an unapproved different customer."
    with pytest.raises(AttributeError):
        stored_context["dependsOn"].append("step_unapproved")
    with pytest.raises(SpecialistPreApprovalError, match="approved context"):
        await call_specialist_with_guard(context, tampered_request, specialist)

    assert specialist.calls == []


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
async def test_store_edited_plan_can_be_approved_without_rerecording_draft() -> None:
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
    store = ApprovalStateStore(agent_descriptors=registry.descriptors())
    store.add_draft(draft_plan)

    edit_result = store.apply_user_action(
        {
            "userAction": {
                "type": "add_instruction",
                "surfaceId": draft_plan.approval_surface_id,
                "payload": {
                    "planId": draft_plan.plan_id,
                    "editedPlanVersion": draft_plan.plan_version,
                    "stepId": "step_internal_knowledge",
                    "instruction": "Prepare risk-focused notes.",
                },
            }
        }
    )
    edited_record = store.get(draft_plan.plan_id)
    approval_result = store.apply_user_action(
        {
            "userAction": {
                "type": "approve_plan",
                "surfaceId": draft_plan.approval_surface_id,
                "payload": {
                    "planId": draft_plan.plan_id,
                    "editedPlanVersion": edit_result.plan_version,
                    "approvedStepIds": [
                        step.step_id for step in edited_record.draft_plan.steps
                    ],
                },
            }
        }
    )
    assert approval_result.approved_plan is not None
    approved_plan = approval_result.approved_plan
    approved_step = _step_for(approved_plan, "internal_knowledge")
    specialist = _FakeSpecialist()
    request = SpecialistRequest(
        request_id="request_edited_approved_plan",
        user_input=approved_step.instruction,
        agent_id=approved_step.agent_id,
        plan_id=approved_plan.plan_id,
        step_id=approved_step.step_id,
        context=_approved_context_for_step(approved_plan, approved_step),
    )

    # Act
    context.mark_plan_approved(
        approved_plan,
        approval_record=store.get(approved_plan.plan_id),
    )
    response = await call_specialist_with_guard(context, request, specialist)

    # Assert
    assert approval_result.status == "approved"
    assert response.agent_id == "internal_knowledge"
    assert specialist.calls == [request]
    assert approved_plan.plan_version == 2
    assert "Additional instruction: Prepare risk-focused notes." in request.user_input


@pytest.mark.asyncio
async def test_mark_plan_approved_rejects_version_bump_without_store_approval() -> None:
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
            "plan_version": draft_plan.plan_version + 1,
            "steps": [step.model_dump() for step in tampered_steps],
        }
    )

    # Act / Assert
    with pytest.raises(PlanApprovalStateError, match="current draft"):
        context.mark_plan_approved(tampered_plan)

    assert context.approved_plan_id is None


@pytest.mark.asyncio
async def test_mark_plan_approved_rejects_mutated_store_approved_copy() -> None:
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
    store = ApprovalStateStore(agent_descriptors=registry.descriptors())
    store.add_draft(draft_plan)

    edit_result = store.apply_user_action(
        {
            "userAction": {
                "type": "add_instruction",
                "surfaceId": draft_plan.approval_surface_id,
                "payload": {
                    "planId": draft_plan.plan_id,
                    "editedPlanVersion": draft_plan.plan_version,
                    "stepId": "step_internal_knowledge",
                    "instruction": "Prepare risk-focused notes.",
                },
            }
        }
    )
    approval_result = store.apply_user_action(
        {
            "userAction": {
                "type": "approve_plan",
                "surfaceId": draft_plan.approval_surface_id,
                "payload": {
                    "planId": draft_plan.plan_id,
                    "editedPlanVersion": edit_result.plan_version,
                    "approvedStepIds": [
                        step.step_id
                        for step in store.get(draft_plan.plan_id).draft_plan.steps
                    ],
                },
            }
        }
    )
    assert approval_result.approved_plan is not None
    tampered_plan = approval_result.approved_plan
    _step_for(tampered_plan, "internal_knowledge").instruction = (
        "Research an unapproved different customer."
    )

    # Act / Assert
    with pytest.raises(PlanApprovalStateError, match="current draft"):
        context.mark_plan_approved(
            tampered_plan,
            approval_record=store.get(draft_plan.plan_id),
        )

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
async def test_specialist_guard_rejects_mismatched_response_agent_owner() -> None:
    # Arrange
    router = RequestRouter(
        slm_client=MockSlmIntentClient(),
        intent_classifier=DeterministicIntentClassifier(),
        registry=AgentRegistry.from_default_config(),
    )
    context = await router.route_request(
        "Summarize the internal notes for ABC Manufacturing."
    )
    calls: list[SpecialistRequest] = []

    async def specialist(request: SpecialistRequest) -> SpecialistResponse:
        calls.append(request)
        return SpecialistResponse(
            response_id="response_mismatched_owner",
            agent_id="credit_risk",
            content="mis-bound specialist response",
            surface_id="surface_mismatched_owner",
        )

    request = SpecialistRequest(
        request_id="request_direct_mismatched_owner",
        user_input=context.user_input,
        agent_id="internal_knowledge",
    )

    # Act / Assert
    with pytest.raises(RuntimeError, match="agent_id must match requested agent_id"):
        await call_specialist_with_guard(context, request, specialist)

    assert calls == [request]
    assert dict(context.specialist_surface_owners) == {}


@pytest.mark.asyncio
async def test_specialist_guard_allows_shipped_direct_route_agent_handler() -> None:
    # Arrange
    router = RequestRouter(
        slm_client=MockSlmIntentClient(),
        intent_classifier=DeterministicIntentClassifier(),
        registry=AgentRegistry.from_default_config(),
    )
    context = await router.route_request(
        "Summarize the internal notes for ABC Manufacturing."
    )
    specialist = build_default_specialists()["internal_knowledge"]
    request = SpecialistRequest(
        request_id="request_direct_shipped_agent",
        user_input=context.user_input,
        agent_id="internal_knowledge",
    )

    # Act
    response = await call_specialist_with_guard(context, request, specialist)

    # Assert
    assert response.agent_id == "internal_knowledge"
    assert specialist.calls == [request]


@pytest.mark.asyncio
async def test_direct_route_records_specialist_a2ui_surface_owner() -> None:
    # Arrange
    router = RequestRouter(
        slm_client=MockSlmIntentClient(),
        intent_classifier=DeterministicIntentClassifier(),
        registry=AgentRegistry.from_default_config(),
    )
    context = await router.route_request(
        "Show product opportunity for ABC Manufacturing."
    )
    specialist = build_default_specialists()["product_opportunity"]
    request = SpecialistRequest(
        request_id="request_direct_product_surface",
        user_input=context.user_input,
        agent_id="product_opportunity",
    )

    # Act
    response = await call_specialist_with_guard(context, request, specialist)

    # Assert
    assert response.surface_id is not None
    assert dict(context.specialist_surface_owners) == {
        response.surface_id: "product_opportunity"
    }
    payload = response.a2ui_payload
    assert isinstance(payload, list)
    update_components = payload[1]["updateComponents"]["components"]
    button_component = next(
        component for component in update_components if component["component"] == "Button"
    )
    user_action = parse_user_action(button_component["action"])
    assert user_action.surface_id == response.surface_id
    assert context.specialist_owner_for_surface(user_action.surface_id) == (
        "product_opportunity"
    )


@pytest.mark.asyncio
async def test_direct_route_derives_surface_owner_from_validated_a2ui_payload() -> None:
    # Arrange
    router = RequestRouter(
        slm_client=MockSlmIntentClient(),
        intent_classifier=DeterministicIntentClassifier(),
        registry=AgentRegistry.from_default_config(),
    )
    context = await router.route_request(
        "Show product opportunity for ABC Manufacturing."
    )
    rendered_surface_id = "surface_rendered_product_action"
    stale_surface_id = "surface_stale_product_action"

    async def specialist(request: SpecialistRequest) -> SpecialistResponse:
        return SpecialistResponse(
            response_id="response_stale_product_surface",
            agent_id=request.agent_id,
            content="Product opportunity response.",
            structured_output={"summary": "Treasury fit identified."},
            a2ui_payload=[
                {
                    "version": "v0.9",
                    "createSurface": {
                        "surfaceId": rendered_surface_id,
                        "catalogId": (
                            "https://a2ui.org/specification/v0_9/"
                            "basic_catalog.json"
                        ),
                    },
                },
                {
                    "version": "v0.9",
                    "updateComponents": {
                        "surfaceId": rendered_surface_id,
                        "components": [
                            {
                                "id": "root",
                                "component": "Card",
                                "child": "component_product_action_content",
                            },
                            {
                                "id": "component_product_action_content",
                                "component": "Column",
                                "children": [
                                    "component_product_action_summary",
                                    "component_product_action_button",
                                ],
                            },
                            {
                                "id": "component_product_action_summary",
                                "component": "Text",
                                "text": "Treasury fit identified.",
                            },
                            {
                                "id": "component_product_action_button",
                                "component": "Button",
                                "child": "component_product_action_button_label",
                                "action": {
                                    "event": {
                                        "name": "specialist_action",
                                        "context": {
                                            "type": "specialist_action",
                                            "surfaceId": rendered_surface_id,
                                            "payload": {
                                                "agentId": request.agent_id,
                                                "action": "show_detail",
                                            },
                                        },
                                    }
                                },
                            },
                            {
                                "id": "component_product_action_button_label",
                                "component": "Text",
                                "text": "Show more detail",
                            },
                        ],
                    },
                },
            ],
            surface_id=stale_surface_id,
        )

    request = SpecialistRequest(
        request_id="request_stale_product_surface",
        user_input=context.user_input,
        agent_id="product_opportunity",
    )

    # Act
    response = await call_specialist_with_guard(context, request, specialist)

    # Assert
    assert response.surface_id == rendered_surface_id
    assert dict(context.specialist_surface_owners) == {
        rendered_surface_id: "product_opportunity"
    }
    assert context.specialist_owner_for_surface(stale_surface_id) is None

    payload = response.a2ui_payload
    assert isinstance(payload, list)
    update_components = payload[1]["updateComponents"]["components"]
    button_component = next(
        component for component in update_components if component["component"] == "Button"
    )
    user_action = parse_user_action(button_component["action"])
    assert user_action.surface_id == rendered_surface_id
    assert context.specialist_owner_for_surface(user_action.surface_id) == (
        "product_opportunity"
    )


@pytest.mark.parametrize(
    ("message_key", "surface_id"),
    [
        ("updateDataModel", "surface_updated_product_model"),
        ("deleteSurface", "surface_deleted_product_model"),
    ],
)
@pytest.mark.asyncio
async def test_direct_route_derives_surface_owner_from_supported_a2ui_surface_messages(
    message_key: str,
    surface_id: str,
) -> None:
    # Arrange
    router = RequestRouter(
        slm_client=MockSlmIntentClient(),
        intent_classifier=DeterministicIntentClassifier(),
        registry=AgentRegistry.from_default_config(),
    )
    context = await router.route_request(
        "Show product opportunity for ABC Manufacturing."
    )

    async def specialist(request: SpecialistRequest) -> SpecialistResponse:
        return SpecialistResponse(
            response_id="response_surface_message_product_action",
            agent_id=request.agent_id,
            content="Product opportunity response.",
            structured_output={"summary": "Treasury fit identified."},
            a2ui_payload={
                "version": "v0.9",
                message_key: {"surfaceId": surface_id},
            },
        )

    request = SpecialistRequest(
        request_id="request_surface_message_product_action",
        user_input=context.user_input,
        agent_id="product_opportunity",
    )

    # Act
    response = await call_specialist_with_guard(context, request, specialist)

    # Assert
    assert response.surface_id == surface_id
    assert response.a2ui_payload == {
        "version": "v0.9",
        message_key: {"surfaceId": surface_id},
    }
    assert dict(context.specialist_surface_owners) == {
        surface_id: "product_opportunity"
    }


@pytest.mark.asyncio
async def test_direct_route_falls_back_for_mixed_surface_a2ui_payload() -> None:
    # Arrange
    router = RequestRouter(
        slm_client=MockSlmIntentClient(),
        intent_classifier=DeterministicIntentClassifier(),
        registry=AgentRegistry.from_default_config(),
    )
    context = await router.route_request(
        "Show product opportunity for ABC Manufacturing."
    )

    async def specialist(request: SpecialistRequest) -> SpecialistResponse:
        return SpecialistResponse(
            response_id="response_mixed_surface_product_action",
            agent_id=request.agent_id,
            content="Product opportunity response.",
            structured_output={"summary": "Treasury fit identified."},
            a2ui_payload=[
                {
                    "version": "v0.9",
                    "updateDataModel": {
                        "surfaceId": "surface_product_action_primary"
                    },
                },
                {
                    "version": "v0.9",
                    "deleteSurface": {
                        "surfaceId": "surface_product_action_secondary"
                    },
                },
            ],
        )

    request = SpecialistRequest(
        request_id="request_mixed_surface_product_action",
        user_input=context.user_input,
        agent_id="product_opportunity",
    )

    # Act
    response = await call_specialist_with_guard(context, request, specialist)

    # Assert
    assert response.a2ui_payload is None
    assert response.surface_id is None
    assert dict(context.specialist_surface_owners) == {}
    diagnostic = response.structured_output["a2ui_validation"]
    assert diagnostic["valid"] is False
    assert "multiple surfaceIds" in repr(diagnostic)


@pytest.mark.asyncio
async def test_direct_route_preserves_structured_specialist_action_surface_owner() -> None:
    # Arrange
    router = RequestRouter(
        slm_client=MockSlmIntentClient(),
        intent_classifier=DeterministicIntentClassifier(),
        registry=AgentRegistry.from_default_config(),
    )
    context = await router.route_request(
        "Show product opportunity for ABC Manufacturing."
    )
    surface_id = "surface_structured_product_action"

    async def specialist(request: SpecialistRequest) -> SpecialistResponse:
        return SpecialistResponse(
            response_id="response_structured_product_action",
            agent_id=request.agent_id,
            content="Product opportunity response.",
            structured_output={"summary": "Treasury fit identified."},
            a2ui_payload=[
                {
                    "version": "v0.9",
                    "createSurface": {
                        "surfaceId": surface_id,
                        "catalogId": (
                            "https://a2ui.org/specification/v0_9/"
                            "basic_catalog.json"
                        ),
                    },
                },
                {
                    "version": "v0.9",
                    "updateComponents": {
                        "surfaceId": surface_id,
                        "components": [
                            {
                                "id": "root",
                                "component": "Card",
                                "child": "component_product_action_content",
                            },
                            {
                                "id": "component_product_action_content",
                                "component": "Column",
                                "children": [
                                    "component_product_action_summary",
                                    "component_product_action_button",
                                ],
                            },
                            {
                                "id": "component_product_action_summary",
                                "component": "Text",
                                "text": "Treasury fit identified.",
                            },
                            {
                                "id": "component_product_action_button",
                                "component": "Button",
                                "child": "component_product_action_button_label",
                                "action": {
                                    "event": {
                                        "name": "specialist_action",
                                        "context": {
                                            "type": "specialist_action",
                                            "surfaceId": surface_id,
                                            "payload": {
                                                "agentId": request.agent_id,
                                                "action": "show_detail",
                                            },
                                        },
                                    }
                                },
                            },
                            {
                                "id": "component_product_action_button_label",
                                "component": "Text",
                                "text": "Show more detail",
                            },
                        ],
                    },
                },
            ],
        )

    request = SpecialistRequest(
        request_id="request_structured_product_surface",
        user_input=context.user_input,
        agent_id="product_opportunity",
    )

    # Act
    response = await call_specialist_with_guard(context, request, specialist)

    # Assert
    assert response.a2ui_payload is not None
    assert response.surface_id == surface_id
    assert dict(context.specialist_surface_owners) == {
        surface_id: "product_opportunity"
    }
    payload = response.a2ui_payload
    assert isinstance(payload, list)
    update_components = payload[1]["updateComponents"]["components"]
    button_component = next(
        component for component in update_components if component["component"] == "Button"
    )
    user_action = parse_user_action(button_component["action"])
    assert user_action.type == "specialist_action"
    assert user_action.surface_id == surface_id
    assert user_action.payload == {
        "agentId": "product_opportunity",
        "action": "show_detail",
    }
    assert context.specialist_owner_for_surface(user_action.surface_id) == (
        "product_opportunity"
    )


@pytest.mark.asyncio
async def test_direct_route_validates_specialist_a2ui_before_returning_response() -> None:
    # Arrange
    router = RequestRouter(
        slm_client=MockSlmIntentClient(),
        intent_classifier=DeterministicIntentClassifier(),
        registry=AgentRegistry.from_default_config(),
    )
    context = await router.route_request(
        "Show product opportunity for ABC Manufacturing."
    )
    leaked_value = "OPENROUTER_API_KEY=sk-live-invalid-a2ui-secret-token-123456789"

    async def specialist(request: SpecialistRequest) -> SpecialistResponse:
        return SpecialistResponse(
            response_id="response_invalid_product_a2ui",
            agent_id=request.agent_id,
            content="Product opportunity response.",
            structured_output={},
            a2ui_payload={
                "version": "v0.9",
                "updateComponents": {
                    "surfaceId": "surface_invalid_product_a2ui",
                    "components": [
                        {
                            "id": "root",
                            "component": "Text",
                            "text": leaked_value,
                            "variant": "body",
                        }
                    ],
                },
            },
            surface_id="surface_invalid_product_a2ui",
        )

    request = SpecialistRequest(
        request_id="request_direct_invalid_product_surface",
        user_input=context.user_input,
        agent_id="product_opportunity",
    )

    # Act
    response = await call_specialist_with_guard(context, request, specialist)

    # Assert
    assert response.a2ui_payload is None
    assert response.surface_id is None
    assert dict(context.specialist_surface_owners) == {}
    diagnostic = response.structured_output["a2ui_validation"]
    diagnostic_text = repr(diagnostic)
    assert diagnostic["valid"] is False
    assert "secret-like value" in diagnostic_text
    assert "<redacted-secret>" in diagnostic_text
    assert leaked_value not in diagnostic_text


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
