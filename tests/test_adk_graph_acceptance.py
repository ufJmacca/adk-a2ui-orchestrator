import sys
from types import ModuleType

import pytest

from google.adk.workflow import Workflow

from orchestrator_demo.contracts import AgentDescriptor, ExecutionPlan, PlanStep
from orchestrator_demo.orchestrator.approval_state import ApprovalStateStore
from orchestrator_demo.orchestrator.graph_builder import GraphBuilder
from orchestrator_demo.orchestrator.graph_runtime import (
    AdkGraphApiError,
    AdkWorkflowRuntimeFactory,
)


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
