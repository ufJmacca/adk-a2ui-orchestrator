import asyncio
import sys
from types import ModuleType

import pytest

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.workflow import Workflow
from google.genai import types

from orchestrator_demo.agents import build_default_specialists
from orchestrator_demo.a2ui_support.event_parser import parse_user_action
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
from orchestrator_demo.orchestrator.approval_state import ApprovalStateStore
from orchestrator_demo.orchestrator.graph_builder import GraphBuilder
from orchestrator_demo.orchestrator.graph_runtime import (
    AdkGraphApiError,
    AdkWorkflowRuntimeFactory,
)
from orchestrator_demo.orchestrator.request_context import RequestContext


def _descriptor(agent_id: str) -> AgentDescriptor:
    return AgentDescriptor(
        agent_id=agent_id,
        display_name=agent_id.replace("_", " ").title(),
        capabilities=["business banking support"],
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        a2ui_catalogs=["basic"],
        routing_examples=[],
        execution_mode="local_llm",
    )


class _Registry:
    def __init__(self, agent_ids: list[str]) -> None:
        self.agent_ids = list(agent_ids)

    def require_plan_agents_available(self, plan: ExecutionPlan) -> None:
        unavailable = [
            agent_id
            for agent_id in plan.selected_agents
            if agent_id not in self.agent_ids
        ]
        if unavailable:
            raise AssertionError(f"unexpected unavailable agents: {unavailable}")


class _RecordingSpecialist:
    def __init__(
        self,
        agent_id: str,
        *,
        content: str | None = None,
        structured_output: dict[str, object] | None = None,
        delay_seconds: float = 0,
    ) -> None:
        self.agent_id = agent_id
        self.content = content
        self.structured_output = structured_output or {}
        self.delay_seconds = delay_seconds
        self.calls: list[SpecialistRequest] = []

    async def handle(self, request: SpecialistRequest) -> SpecialistResponse:
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        self.calls.append(request)
        structured_output = {
            "agent_id": self.agent_id,
            "request_id": request.request_id,
            **self.structured_output,
        }
        return SpecialistResponse(
            response_id=f"response_{request.request_id.removeprefix('request_')}",
            agent_id=self.agent_id,
            content=self.content or f"{self.agent_id} completed {request.step_id}",
            structured_output=structured_output,
        )


def _approved_record(plan: ExecutionPlan):
    store = ApprovalStateStore(
        agent_descriptors=[_descriptor(agent_id) for agent_id in plan.selected_agents]
    )
    store.add_draft(plan)
    result = store.apply_user_action(
        {
            "userAction": {
                "type": "approve_plan",
                "surfaceId": plan.approval_surface_id,
                "payload": {
                    "planId": plan.plan_id,
                    "approvedStepIds": [step.step_id for step in plan.steps],
                    "editedPlanVersion": plan.plan_version,
                },
            }
        }
    )
    assert result.status == "approved"
    return store.get(plan.plan_id)


def _approved_complex_plan() -> ExecutionPlan:
    return ExecutionPlan(
        plan_id="plan_acceptance_adk_graph",
        objective="Research this prospect and give me risks and opportunities.",
        detected_intents=["prospect_research"],
        selected_agents=[
            "web_search",
            "industry_research",
            "product_opportunity",
            "credit_risk",
            "synthesis",
        ],
        steps=[
            PlanStep(
                step_id="step_web_search",
                agent_id="web_search",
                instruction="Gather public information about the prospect.",
                expected_output="Public research summary.",
                data_source_categories=["public_web"],
                parallel_group="parallel_prospect_research",
            ),
            PlanStep(
                step_id="step_industry_research",
                agent_id="industry_research",
                instruction="Assess industry context.",
                expected_output="Industry risks and opportunities.",
                data_source_categories=["industry_research"],
                parallel_group="parallel_prospect_research",
            ),
            PlanStep(
                step_id="step_product_opportunity",
                agent_id="product_opportunity",
                instruction="Identify product opportunities.",
                expected_output="Product opportunity shortlist.",
                data_source_categories=["product_fit"],
                parallel_group="parallel_prospect_research",
            ),
            PlanStep(
                step_id="step_credit_risk",
                agent_id="credit_risk",
                instruction="Flag credit risk themes.",
                expected_output="Credit risk themes.",
                data_source_categories=["credit_risk"],
                parallel_group="parallel_prospect_research",
            ),
            PlanStep(
                step_id="step_synthesis",
                agent_id="synthesis",
                instruction="Combine specialist outputs into an RM-ready answer.",
                depends_on=[
                    "step_web_search",
                    "step_industry_research",
                    "step_product_opportunity",
                    "step_credit_risk",
                ],
                expected_output="Final prospect briefing.",
                data_source_categories=["specialist_outputs"],
            ),
        ],
        data_source_categories=[
            "public_web",
            "industry_research",
            "product_fit",
            "credit_risk",
        ],
        risk_notes=["Synthetic demo data only."],
        approval_surface_id="surface_plan_acceptance_adk_graph",
    )


def _conditional_data_quality_plan() -> ExecutionPlan:
    return ExecutionPlan(
        plan_id="plan_acceptance_conditional_data_quality",
        objective="Prepare a briefing and check missing internal data.",
        detected_intents=["meeting_prep"],
        selected_agents=["internal_knowledge", "data_quality", "synthesis"],
        steps=[
            PlanStep(
                step_id="step_internal_knowledge",
                agent_id="internal_knowledge",
                instruction="Review internal CRM notes for missing data.",
                expected_output="Internal customer context.",
                data_source_categories=["internal_crm"],
            ),
            PlanStep(
                step_id="step_data_quality",
                agent_id="data_quality",
                instruction="Check missing or stale internal data if needed.",
                depends_on=["step_internal_knowledge"],
                condition="missing_internal_data",
                expected_output="Data quality gaps.",
                data_source_categories=["data_quality"],
            ),
            PlanStep(
                step_id="step_synthesis",
                agent_id="synthesis",
                instruction="Synthesize the available context.",
                depends_on=["step_data_quality"],
                expected_output="Final briefing or clarification need.",
                data_source_categories=["specialist_outputs"],
            ),
        ],
        data_source_categories=["internal_crm", "data_quality"],
        risk_notes=["Synthetic demo data only."],
        approval_surface_id="surface_plan_acceptance_conditional_data_quality",
    )


def _generic_conditional_review_plan() -> ExecutionPlan:
    return ExecutionPlan(
        plan_id="plan_acceptance_generic_conditional_review",
        objective="Prepare a review only if the internal context asks for it.",
        detected_intents=["meeting_prep"],
        selected_agents=["internal_knowledge", "credit_risk"],
        steps=[
            PlanStep(
                step_id="step_internal_knowledge",
                agent_id="internal_knowledge",
                instruction="Review internal CRM notes for review triggers.",
                expected_output="Internal customer context.",
                data_source_categories=["internal_crm"],
            ),
            PlanStep(
                step_id="step_credit_risk",
                agent_id="credit_risk",
                instruction="Assess credit risk if review is needed.",
                depends_on=["step_internal_knowledge"],
                condition="needs_review",
                expected_output="Conditional credit risk themes.",
                data_source_categories=["credit_risk"],
            ),
        ],
        data_source_categories=["internal_crm", "credit_risk"],
        risk_notes=["Synthetic demo data only."],
        approval_surface_id="surface_plan_acceptance_generic_conditional_review",
    )


def _generic_conditional_fan_in_plan() -> ExecutionPlan:
    return ExecutionPlan(
        plan_id="plan_acceptance_generic_conditional_fan_in",
        objective="Prepare a review with internal and industry prerequisites.",
        detected_intents=["meeting_prep"],
        selected_agents=["internal_knowledge", "industry_research", "credit_risk"],
        steps=[
            PlanStep(
                step_id="step_internal_knowledge",
                agent_id="internal_knowledge",
                instruction="Review internal CRM notes for review triggers.",
                expected_output="Internal customer context.",
                data_source_categories=["internal_crm"],
            ),
            PlanStep(
                step_id="step_industry_research",
                agent_id="industry_research",
                instruction="Gather industry context.",
                expected_output="Industry context.",
                data_source_categories=["industry_research"],
            ),
            PlanStep(
                step_id="step_credit_risk",
                agent_id="credit_risk",
                instruction="Assess credit risk if review is needed.",
                depends_on=["step_internal_knowledge", "step_industry_research"],
                condition="needs_review",
                expected_output="Conditional credit risk themes.",
                data_source_categories=["credit_risk"],
            ),
        ],
        data_source_categories=["internal_crm", "industry_research", "credit_risk"],
        risk_notes=["Synthetic demo data only."],
        approval_surface_id="surface_plan_acceptance_generic_conditional_fan_in",
    )


def _conditional_data_quality_with_shared_synthesis_plan() -> ExecutionPlan:
    return ExecutionPlan(
        plan_id="plan_acceptance_conditional_shared_synthesis",
        objective="Prepare a briefing with shared industry context.",
        detected_intents=["meeting_prep"],
        selected_agents=[
            "internal_knowledge",
            "industry_research",
            "data_quality",
            "synthesis",
        ],
        steps=[
            PlanStep(
                step_id="step_internal_knowledge",
                agent_id="internal_knowledge",
                instruction="Review internal CRM notes for missing data.",
                expected_output="Internal customer context.",
                data_source_categories=["internal_crm"],
            ),
            PlanStep(
                step_id="step_industry_research",
                agent_id="industry_research",
                instruction="Gather industry context.",
                expected_output="Industry context.",
                data_source_categories=["industry_research"],
            ),
            PlanStep(
                step_id="step_data_quality",
                agent_id="data_quality",
                instruction="Check missing or stale internal data if needed.",
                depends_on=["step_internal_knowledge"],
                condition="missing_internal_data",
                expected_output="Data quality gaps.",
                data_source_categories=["data_quality"],
            ),
            PlanStep(
                step_id="step_synthesis",
                agent_id="synthesis",
                instruction="Synthesize data quality and industry context.",
                depends_on=["step_data_quality", "step_industry_research"],
                expected_output="Final briefing or clarification need.",
                data_source_categories=["specialist_outputs"],
            ),
        ],
        data_source_categories=["internal_crm", "industry_research", "data_quality"],
        risk_notes=["Synthetic demo data only."],
        approval_surface_id="surface_plan_acceptance_conditional_shared_synthesis",
    )


def _conditional_data_quality_with_route_gated_synthesis_plan() -> ExecutionPlan:
    return ExecutionPlan(
        plan_id="plan_acceptance_conditional_route_gated_synthesis",
        objective="Prepare a synthesis after the selected data-quality route.",
        detected_intents=["meeting_prep"],
        selected_agents=[
            "internal_knowledge",
            "industry_research",
            "data_quality",
            "compliance_policy",
            "synthesis",
        ],
        steps=[
            PlanStep(
                step_id="step_internal_knowledge",
                agent_id="internal_knowledge",
                instruction="Review internal CRM notes and choose a follow-up route.",
                expected_output="Internal customer context and selected route.",
                data_source_categories=["internal_crm"],
            ),
            PlanStep(
                step_id="step_industry_research",
                agent_id="industry_research",
                instruction="Gather industry context required for every synthesis.",
                expected_output="Industry context.",
                data_source_categories=["industry_research"],
            ),
            PlanStep(
                step_id="step_data_quality",
                agent_id="data_quality",
                instruction="Check missing or stale internal data if needed.",
                depends_on=["step_internal_knowledge"],
                condition="missing_internal_data",
                expected_output="Data quality gaps.",
                data_source_categories=["data_quality"],
            ),
            PlanStep(
                step_id="step_compliance_policy",
                agent_id="compliance_policy",
                instruction="Review policy only when policy review is selected.",
                depends_on=["step_internal_knowledge"],
                condition="policy_review",
                expected_output="Policy caveats.",
                data_source_categories=["compliance_policy"],
            ),
            PlanStep(
                step_id="step_synthesis",
                agent_id="synthesis",
                instruction="Synthesize the selected conditional path and industry context.",
                depends_on=[
                    "step_data_quality",
                    "step_compliance_policy",
                    "step_industry_research",
                ],
                expected_output="Final briefing.",
                data_source_categories=["specialist_outputs"],
            ),
        ],
        data_source_categories=[
            "internal_crm",
            "industry_research",
            "data_quality",
            "compliance_policy",
        ],
        risk_notes=["Synthetic demo data only."],
        approval_surface_id="surface_plan_acceptance_conditional_route_gated_synthesis",
    )


def _mutually_exclusive_conditional_synthesis_plan() -> ExecutionPlan:
    return ExecutionPlan(
        plan_id="plan_acceptance_mutually_exclusive_conditional_synthesis",
        objective="Prepare a synthesis after the selected conditional review path.",
        detected_intents=["meeting_prep"],
        selected_agents=[
            "internal_knowledge",
            "industry_research",
            "credit_risk",
            "compliance_policy",
            "synthesis",
        ],
        steps=[
            PlanStep(
                step_id="step_internal_knowledge",
                agent_id="internal_knowledge",
                instruction="Review internal CRM notes and choose the review path.",
                expected_output="Internal customer context and selected review path.",
                data_source_categories=["internal_crm"],
            ),
            PlanStep(
                step_id="step_industry_research",
                agent_id="industry_research",
                instruction="Gather industry context required for every synthesis.",
                expected_output="Industry context.",
                data_source_categories=["industry_research"],
            ),
            PlanStep(
                step_id="step_credit_risk",
                agent_id="credit_risk",
                instruction="Assess credit risk only when risk review is selected.",
                depends_on=["step_internal_knowledge"],
                condition="needs_review",
                expected_output="Conditional credit risk themes.",
                data_source_categories=["credit_risk"],
            ),
            PlanStep(
                step_id="step_compliance_policy",
                agent_id="compliance_policy",
                instruction="Review policy only when policy review is selected.",
                depends_on=["step_internal_knowledge"],
                condition="policy_review",
                expected_output="Conditional policy caveats.",
                data_source_categories=["compliance_policy"],
            ),
            PlanStep(
                step_id="step_synthesis",
                agent_id="synthesis",
                instruction="Synthesize the selected review path and industry context.",
                depends_on=[
                    "step_credit_risk",
                    "step_compliance_policy",
                    "step_industry_research",
                ],
                expected_output="Final briefing.",
                data_source_categories=["specialist_outputs"],
            ),
        ],
        data_source_categories=[
            "internal_crm",
            "industry_research",
            "credit_risk",
            "compliance_policy",
        ],
        risk_notes=["Synthetic demo data only."],
        approval_surface_id=(
            "surface_plan_acceptance_mutually_exclusive_conditional_synthesis"
        ),
    )


async def _run_workflow_events(workflow: Workflow):
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name="test_graph_acceptance",
        user_id="relationship_manager",
    )
    runner = Runner(
        app_name="test_graph_acceptance",
        node=workflow,
        session_service=session_service,
    )
    events = []
    async for event in runner.run_async(
        user_id="relationship_manager",
        session_id=session.id,
        new_message=types.Content(
            role="user",
            parts=[types.Part(text="run the approved plan")],
        ),
    ):
        events.append(event)
    return events


def test_acceptance_uses_adk_workflow_runtime_for_approved_complex_plan() -> None:
    # Arrange
    plan = _approved_complex_plan()
    builder = GraphBuilder(registry=_Registry(plan.selected_agents))

    # Act
    result = builder.build(_approved_record(plan))

    # Assert
    assert result.runtime.is_adk_backed is True
    assert isinstance(result.runtime.workflow, Workflow)
    assert result.runtime.workflow.graph is not None
    assert result.runtime.workflow.graph.nodes
    assert result.runtime.workflow.graph.edges
    assert result.spec.pattern == "fan_out_fan_in"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "structured_output",
    [
        {"missing_internal_data": True},
        {"selected_route": "missing_internal_data"},
    ],
)
async def test_acceptance_workflow_executes_specialists_and_routes_data_quality_branch(
    structured_output: dict[str, object],
) -> None:
    # Arrange
    plan = _conditional_data_quality_plan()
    specialists = {
        "internal_knowledge": _RecordingSpecialist(
            "internal_knowledge",
            structured_output=structured_output,
        ),
        "data_quality": _RecordingSpecialist("data_quality"),
        "synthesis": _RecordingSpecialist("synthesis"),
    }
    runtime_factory = AdkWorkflowRuntimeFactory(specialists=specialists)
    builder = GraphBuilder(
        registry=_Registry(plan.selected_agents),
        runtime_factory=runtime_factory,
    )

    # Act
    result = builder.build(_approved_record(plan))
    events = await _run_workflow_events(result.runtime.workflow)

    # Assert
    assert specialists["internal_knowledge"].calls[0].user_input == (
        "Review internal CRM notes for missing data."
    )
    assert specialists["data_quality"].calls[0].context["expectedOutput"] == (
        "Data quality gaps."
    )
    assert (
        specialists["data_quality"].calls[0].context["upstream"]["response"][
            "agent_id"
        ]
        == "internal_knowledge"
    )
    assert len(specialists["data_quality"].calls) == 1
    assert len(specialists["synthesis"].calls) == 1
    assert (
        specialists["synthesis"].calls[0].context["upstream"]["response"][
            "agent_id"
        ]
        == "data_quality"
    )
    assert "missing_internal_data" in [
        getattr(event.actions, "route", None) for event in events
    ]
    assert any(
        event.output
        and event.output["response"]["agent_id"] == "data_quality"
        for event in events
    )


@pytest.mark.asyncio
async def test_acceptance_conditional_route_uses_specialist_output_not_prompt_text() -> None:
    # Arrange
    plan = _conditional_data_quality_plan()
    specialists = {
        "internal_knowledge": _RecordingSpecialist(
            "internal_knowledge",
            content="No missing data found.",
            structured_output={"missing_internal_data": False},
        ),
        "data_quality": _RecordingSpecialist("data_quality"),
        "synthesis": _RecordingSpecialist("synthesis"),
    }
    runtime_factory = AdkWorkflowRuntimeFactory(specialists=specialists)
    builder = GraphBuilder(
        registry=_Registry(plan.selected_agents),
        runtime_factory=runtime_factory,
    )

    # Act
    result = builder.build(_approved_record(plan))
    events = await _run_workflow_events(result.runtime.workflow)

    # Assert
    assert len(specialists["data_quality"].calls) == 0
    assert len(specialists["synthesis"].calls) == 1
    assert (
        specialists["synthesis"].calls[0].context["upstream"]["response"][
            "agent_id"
        ]
        == "internal_knowledge"
    )
    assert "__DEFAULT__" in [
        getattr(event.actions, "route", None) for event in events
    ]


@pytest.mark.asyncio
async def test_acceptance_conditional_route_treats_negative_string_flag_as_default() -> None:
    # Arrange
    plan = _conditional_data_quality_plan()
    specialists = {
        "internal_knowledge": _RecordingSpecialist(
            "internal_knowledge",
            structured_output={"missing_internal_data": "no data quality issues"},
        ),
        "data_quality": _RecordingSpecialist("data_quality"),
        "synthesis": _RecordingSpecialist("synthesis"),
    }
    runtime_factory = AdkWorkflowRuntimeFactory(specialists=specialists)
    builder = GraphBuilder(
        registry=_Registry(plan.selected_agents),
        runtime_factory=runtime_factory,
    )

    # Act
    result = builder.build(_approved_record(plan))
    events = await _run_workflow_events(result.runtime.workflow)

    # Assert
    assert len(specialists["data_quality"].calls) == 0
    assert len(specialists["synthesis"].calls) == 1
    assert (
        specialists["synthesis"].calls[0].context["upstream"]["response"][
            "agent_id"
        ]
        == "internal_knowledge"
    )
    assert "__DEFAULT__" in [
        getattr(event.actions, "route", None) for event in events
    ]
    assert "missing_internal_data" not in [
        getattr(event.actions, "route", None) for event in events
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("structured_output", "expected_credit_risk_calls"),
    [
        ({"needs_review": False}, 0),
        ({"selected_route": "needs_review"}, 1),
    ],
)
async def test_acceptance_generic_conditional_route_requires_explicit_output_signal(
    structured_output: dict[str, object],
    expected_credit_risk_calls: int,
) -> None:
    # Arrange
    plan = _generic_conditional_review_plan()
    specialists = {
        "internal_knowledge": _RecordingSpecialist(
            "internal_knowledge",
            structured_output=structured_output,
        ),
        "credit_risk": _RecordingSpecialist("credit_risk"),
    }
    runtime_factory = AdkWorkflowRuntimeFactory(specialists=specialists)
    builder = GraphBuilder(
        registry=_Registry(plan.selected_agents),
        runtime_factory=runtime_factory,
    )

    # Act
    result = builder.build(_approved_record(plan))
    events = await _run_workflow_events(result.runtime.workflow)

    # Assert
    assert len(specialists["credit_risk"].calls) == expected_credit_risk_calls
    route_events = [getattr(event.actions, "route", None) for event in events]
    if expected_credit_risk_calls:
        assert "needs_review" in route_events
        assert any(
            event.output
            and event.output["response"]["agent_id"] == "credit_risk"
            for event in events
        )
    else:
        assert "needs_review" not in route_events
        assert not any(
            event.output
            and event.output["response"]["agent_id"] == "credit_risk"
            for event in events
        )


@pytest.mark.asyncio
async def test_acceptance_conditional_fan_in_skips_unselected_branch() -> None:
    # Arrange
    plan = _generic_conditional_fan_in_plan()
    specialists = {
        "internal_knowledge": _RecordingSpecialist(
            "internal_knowledge",
            structured_output={"needs_review": False},
        ),
        "industry_research": _RecordingSpecialist(
            "industry_research",
            delay_seconds=0.01,
        ),
        "credit_risk": _RecordingSpecialist("credit_risk"),
    }
    runtime_factory = AdkWorkflowRuntimeFactory(specialists=specialists)
    builder = GraphBuilder(
        registry=_Registry(plan.selected_agents),
        runtime_factory=runtime_factory,
    )

    # Act
    result = builder.build(_approved_record(plan))
    events = await _run_workflow_events(result.runtime.workflow)

    # Assert
    assert len(specialists["internal_knowledge"].calls) == 1
    assert len(specialists["industry_research"].calls) == 1
    assert len(specialists["credit_risk"].calls) == 0
    assert "needs_review" not in [
        getattr(event.actions, "route", None) for event in events
    ]
    assert not any(
        event.output and event.output["response"]["agent_id"] == "credit_risk"
        for event in events
    )


@pytest.mark.asyncio
async def test_acceptance_conditional_fan_in_waits_for_unconditional_dependency() -> None:
    # Arrange
    plan = _generic_conditional_fan_in_plan()
    specialists = {
        "internal_knowledge": _RecordingSpecialist(
            "internal_knowledge",
            structured_output={"selected_route": "needs_review"},
        ),
        "industry_research": _RecordingSpecialist(
            "industry_research",
            delay_seconds=0.01,
        ),
        "credit_risk": _RecordingSpecialist("credit_risk"),
    }
    runtime_factory = AdkWorkflowRuntimeFactory(specialists=specialists)
    builder = GraphBuilder(
        registry=_Registry(plan.selected_agents),
        runtime_factory=runtime_factory,
    )

    # Act
    result = builder.build(_approved_record(plan))
    await _run_workflow_events(result.runtime.workflow)

    # Assert
    assert len(specialists["internal_knowledge"].calls) == 1
    assert len(specialists["industry_research"].calls) == 1
    assert len(specialists["credit_risk"].calls) == 1
    upstream = specialists["credit_risk"].calls[0].context["upstream"]
    assert upstream["graph_step_internal_knowledge"]["response"]["agent_id"] == (
        "internal_knowledge"
    )
    assert upstream["graph_step_industry_research"]["response"]["agent_id"] == (
        "industry_research"
    )


@pytest.mark.asyncio
async def test_acceptance_conditional_shared_synthesis_runs_once_after_selected_path() -> None:
    # Arrange
    plan = _conditional_data_quality_with_shared_synthesis_plan()
    specialists = {
        "internal_knowledge": _RecordingSpecialist(
            "internal_knowledge",
            structured_output={"selected_route": "missing_internal_data"},
        ),
        "industry_research": _RecordingSpecialist(
            "industry_research",
            delay_seconds=0.01,
        ),
        "data_quality": _RecordingSpecialist(
            "data_quality",
            delay_seconds=0.03,
        ),
        "synthesis": _RecordingSpecialist("synthesis"),
    }
    runtime_factory = AdkWorkflowRuntimeFactory(specialists=specialists)
    builder = GraphBuilder(
        registry=_Registry(plan.selected_agents),
        runtime_factory=runtime_factory,
    )

    # Act
    result = builder.build(_approved_record(plan))
    await _run_workflow_events(result.runtime.workflow)

    # Assert
    assert len(specialists["industry_research"].calls) == 1
    assert len(specialists["data_quality"].calls) == 1
    assert len(specialists["synthesis"].calls) == 1
    upstream = specialists["synthesis"].calls[0].context["upstream"]
    assert upstream["graph_step_data_quality"]["response"]["agent_id"] == (
        "data_quality"
    )
    assert upstream["graph_step_industry_research"]["response"]["agent_id"] == (
        "industry_research"
    )


@pytest.mark.asyncio
async def test_acceptance_data_quality_synthesis_ignores_unselected_route_gated_branch() -> None:
    # Arrange
    plan = _conditional_data_quality_with_route_gated_synthesis_plan()
    specialists = {
        "internal_knowledge": _RecordingSpecialist(
            "internal_knowledge",
            structured_output={"selected_route": "missing_internal_data"},
        ),
        "industry_research": _RecordingSpecialist(
            "industry_research",
            delay_seconds=0.01,
        ),
        "data_quality": _RecordingSpecialist("data_quality"),
        "compliance_policy": _RecordingSpecialist("compliance_policy"),
        "synthesis": _RecordingSpecialist("synthesis"),
    }
    runtime_factory = AdkWorkflowRuntimeFactory(specialists=specialists)
    builder = GraphBuilder(
        registry=_Registry(plan.selected_agents),
        runtime_factory=runtime_factory,
    )

    # Act
    result = builder.build(_approved_record(plan))
    await _run_workflow_events(result.runtime.workflow)

    # Assert
    assert len(specialists["data_quality"].calls) == 1
    assert len(specialists["industry_research"].calls) == 1
    assert len(specialists["compliance_policy"].calls) == 0
    assert len(specialists["synthesis"].calls) == 1
    upstream = specialists["synthesis"].calls[0].context["upstream"]
    assert upstream["graph_step_data_quality"]["response"]["agent_id"] == (
        "data_quality"
    )
    assert upstream["graph_step_industry_research"]["response"]["agent_id"] == (
        "industry_research"
    )
    assert "graph_step_compliance_policy" not in upstream


@pytest.mark.asyncio
async def test_acceptance_mutually_exclusive_conditional_fan_in_reaches_synthesis() -> None:
    # Arrange
    plan = _mutually_exclusive_conditional_synthesis_plan()
    specialists = {
        "internal_knowledge": _RecordingSpecialist(
            "internal_knowledge",
            structured_output={"selected_route": "policy_review"},
        ),
        "industry_research": _RecordingSpecialist(
            "industry_research",
            delay_seconds=0.01,
        ),
        "credit_risk": _RecordingSpecialist("credit_risk"),
        "compliance_policy": _RecordingSpecialist("compliance_policy"),
        "synthesis": _RecordingSpecialist("synthesis"),
    }
    runtime_factory = AdkWorkflowRuntimeFactory(specialists=specialists)
    builder = GraphBuilder(
        registry=_Registry(plan.selected_agents),
        runtime_factory=runtime_factory,
    )

    # Act
    result = builder.build(_approved_record(plan))
    await _run_workflow_events(result.runtime.workflow)

    # Assert
    assert "join_graph_step_synthesis" not in result.runtime.node_names
    assert "join_graph_step_synthesis_policy_review" in result.runtime.node_names
    assert len(specialists["credit_risk"].calls) == 0
    assert len(specialists["compliance_policy"].calls) == 1
    assert len(specialists["industry_research"].calls) == 1
    assert len(specialists["synthesis"].calls) == 1
    upstream = specialists["synthesis"].calls[0].context["upstream"]
    assert "graph_step_credit_risk" not in upstream
    assert upstream["graph_step_compliance_policy"]["response"]["agent_id"] == (
        "compliance_policy"
    )
    assert upstream["graph_step_industry_research"]["response"]["agent_id"] == (
        "industry_research"
    )


@pytest.mark.asyncio
async def test_acceptance_graph_records_a2ui_surface_owner_for_followup_event() -> None:
    # Arrange
    plan = _approved_complex_plan()
    context = RequestContext(
        user_input=plan.objective,
        slm_suggestion=IntentSuggestion(intent="prospect_research", confidence=0.7),
        llm_assessment=LlmIntentAssessment(
            intents=["prospect_research"],
            confidence=0.8,
            complexity="complex",
            rationale="Graph acceptance test.",
            required_agents=plan.selected_agents,
        ),
        decision=RoutingDecision(
            path="plan_required",
            selected_agent=None,
            confidence=0.8,
            reason="Graph acceptance test.",
        ),
    )
    runtime_factory = AdkWorkflowRuntimeFactory(
        specialists=build_default_specialists(),
        record_specialist_surface_owner=context.record_specialist_surface_owner,
    )
    builder = GraphBuilder(
        registry=_Registry(plan.selected_agents),
        runtime_factory=runtime_factory,
    )

    # Act
    result = builder.build(_approved_record(plan))
    events = await _run_workflow_events(result.runtime.workflow)

    # Assert
    product_output = next(
        event.output
        for event in events
        if event.output
        and event.output["response"]["agent_id"] == "product_opportunity"
    )
    response = product_output["response"]
    surface_id = response["surface_id"]
    assert context.specialist_owner_for_surface(surface_id) == "product_opportunity"

    payload = response["a2ui_payload"]
    assert isinstance(payload, list)
    update_components = payload[1]["updateComponents"]["components"]
    button_component = next(
        component for component in update_components if component["component"] == "Button"
    )
    user_action = parse_user_action(button_component["action"])
    assert user_action.surface_id == surface_id
    assert context.specialist_owner_for_surface(user_action.surface_id) == (
        "product_opportunity"
    )


def test_acceptance_reports_clear_error_when_adk_workflow_api_is_unavailable() -> None:
    # Arrange
    plan = _approved_complex_plan()
    runtime_factory = AdkWorkflowRuntimeFactory(
        workflow_module="not_a_real_google_adk_workflow_module"
    )
    builder = GraphBuilder(
        registry=_Registry(plan.selected_agents),
        runtime_factory=runtime_factory,
    )

    # Act / Assert
    with pytest.raises(
        AdkGraphApiError,
        match="ADK workflow graph API unavailable or incompatible",
    ):
        builder.build(_approved_record(plan))


def test_acceptance_reports_clear_error_when_importable_adk_api_is_incompatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    class FakeFunctionNode:
        def __init__(self, *, func, name: str) -> None:
            self.func = func
            self.name = name

    class FakeJoinNode:
        def __init__(self, *, name: str) -> None:
            self.name = name

    class FakeEdge:
        def __init__(self, *, from_node, to_node, route=None) -> None:
            self.from_node = from_node
            self.to_node = to_node
            self.route = route

    class FakeWorkflow:
        def __init__(self, *, name: str, edges: list[FakeEdge]) -> None:
            self.name = name
            self.edges = edges
            self.graph = object()

    fake_workflow_module = ModuleType("fake_incompatible_adk_workflow")
    fake_workflow_module.Workflow = FakeWorkflow
    fake_workflow_module.FunctionNode = FakeFunctionNode
    fake_workflow_module.JoinNode = FakeJoinNode
    fake_workflow_module.Edge = FakeEdge
    fake_workflow_module.START = FakeJoinNode(name="__START__")
    monkeypatch.setitem(sys.modules, fake_workflow_module.__name__, fake_workflow_module)

    plan = _approved_complex_plan()
    runtime_factory = AdkWorkflowRuntimeFactory(
        workflow_module=fake_workflow_module.__name__
    )
    builder = GraphBuilder(
        registry=_Registry(plan.selected_agents),
        runtime_factory=runtime_factory,
    )

    # Act / Assert
    with pytest.raises(
        AdkGraphApiError,
        match="ADK workflow graph API unavailable or incompatible",
    ):
        builder.build(_approved_record(plan))
