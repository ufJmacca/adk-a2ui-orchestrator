import pytest

from orchestrator_demo.contracts import AgentDescriptor, ExecutionPlan, PlanStep
from orchestrator_demo.orchestrator.approval_state import (
    ApprovalRecord,
    ApprovalStateStore,
)
from orchestrator_demo.orchestrator.graph_builder import (
    GraphBuilder,
    GraphPlanApprovalError,
)
from orchestrator_demo.registry.agent_registry import UnavailableAgentError


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
        self.checked_plans: list[str] = []

    def require_plan_agents_available(self, plan: ExecutionPlan) -> None:
        self.checked_plans.append(plan.plan_id)
        unavailable = [
            agent_id
            for agent_id in plan.selected_agents
            if agent_id not in self.agent_ids
        ]
        if unavailable:
            raise UnavailableAgentError(
                f"approved plan {plan.plan_id} references unavailable agents: "
                f"{', '.join(unavailable)}"
            )


def _registry_for(plan: ExecutionPlan) -> _Registry:
    return _Registry(list(plan.selected_agents))


def _approval_record_for(plan: ExecutionPlan):
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
    assert result.approved_plan is not None
    return store.get(plan.plan_id)


def _approved_record_without_store_validation(plan: ExecutionPlan) -> ApprovalRecord:
    record = ApprovalRecord(
        draft_plan=plan.model_copy(deep=True),
        status="approved",
        approved_version=plan.plan_version,
    )
    record.approved_plan = plan.model_copy(deep=True)
    return record


def _plan(
    *,
    plan_id: str,
    selected_agents: list[str],
    steps: list[PlanStep],
    immutable_after_approval: bool = True,
) -> ExecutionPlan:
    return ExecutionPlan(
        plan_id=plan_id,
        objective="Prepare a synthetic business banking work product.",
        detected_intents=["meeting_prep"],
        selected_agents=selected_agents,
        steps=steps,
        data_source_categories=["internal_crm"],
        risk_notes=["Synthetic demo data only."],
        approval_surface_id=f"surface_{plan_id}",
        immutable_after_approval=immutable_after_approval,
    )


def _direct_plan() -> ExecutionPlan:
    return _plan(
        plan_id="plan_direct_route",
        selected_agents=["internal_knowledge"],
        steps=[
            PlanStep(
                step_id="step_internal_knowledge",
                agent_id="internal_knowledge",
                instruction="Summarize internal notes.",
                expected_output="Internal customer context.",
                data_source_categories=["internal_crm"],
            )
        ],
    )


def _sequential_plan() -> ExecutionPlan:
    return _plan(
        plan_id="plan_sequential_review",
        selected_agents=["internal_knowledge", "credit_risk", "synthesis"],
        steps=[
            PlanStep(
                step_id="step_internal_knowledge",
                agent_id="internal_knowledge",
                instruction="Review internal CRM notes.",
                expected_output="Internal customer context.",
                data_source_categories=["internal_crm"],
            ),
            PlanStep(
                step_id="step_credit_risk",
                agent_id="credit_risk",
                instruction="Assess credit risk themes from internal context.",
                depends_on=["step_internal_knowledge"],
                expected_output="Credit risk themes.",
                data_source_categories=["credit_risk"],
            ),
            PlanStep(
                step_id="step_synthesis",
                agent_id="synthesis",
                instruction="Synthesize the sequential review.",
                depends_on=["step_credit_risk"],
                expected_output="Final RM-ready review.",
                data_source_categories=["specialist_outputs"],
            ),
        ],
    )


def _fan_out_fan_in_plan() -> ExecutionPlan:
    return _plan(
        plan_id="plan_parallel_meeting_prep",
        selected_agents=[
            "relationship_summary",
            "internal_knowledge",
            "industry_research",
            "synthesis",
        ],
        steps=[
            PlanStep(
                step_id="step_relationship_summary",
                agent_id="relationship_summary",
                instruction="Summarize relationship history.",
                expected_output="Relationship summary.",
                data_source_categories=["relationship_history"],
                parallel_group="parallel_meeting_context",
            ),
            PlanStep(
                step_id="step_internal_knowledge",
                agent_id="internal_knowledge",
                instruction="Review internal CRM notes.",
                expected_output="Internal customer context.",
                data_source_categories=["internal_crm"],
                parallel_group="parallel_meeting_context",
            ),
            PlanStep(
                step_id="step_industry_research",
                agent_id="industry_research",
                instruction="Gather industry context.",
                expected_output="Industry risks and opportunities.",
                data_source_categories=["industry_research"],
                parallel_group="parallel_meeting_context",
            ),
            PlanStep(
                step_id="step_synthesis",
                agent_id="synthesis",
                instruction="Create the RM-ready meeting brief.",
                depends_on=[
                    "step_relationship_summary",
                    "step_internal_knowledge",
                    "step_industry_research",
                ],
                expected_output="Final meeting preparation brief.",
                data_source_categories=["specialist_outputs"],
            ),
        ],
    )


def _mixed_plan() -> ExecutionPlan:
    return _plan(
        plan_id="plan_mixed_risk_review",
        selected_agents=[
            "internal_knowledge",
            "credit_risk",
            "compliance_policy",
            "synthesis",
        ],
        steps=[
            PlanStep(
                step_id="step_internal_knowledge",
                agent_id="internal_knowledge",
                instruction="Review internal CRM notes.",
                expected_output="Internal customer context.",
                data_source_categories=["internal_crm"],
            ),
            PlanStep(
                step_id="step_credit_risk",
                agent_id="credit_risk",
                instruction="Assess credit risk themes.",
                depends_on=["step_internal_knowledge"],
                expected_output="Credit risk themes.",
                data_source_categories=["credit_risk"],
                parallel_group="parallel_risk_policy_review",
            ),
            PlanStep(
                step_id="step_compliance_policy",
                agent_id="compliance_policy",
                instruction="Review policy caveats.",
                depends_on=["step_internal_knowledge"],
                expected_output="Policy caveats.",
                data_source_categories=["compliance_policy"],
                parallel_group="parallel_risk_policy_review",
            ),
            PlanStep(
                step_id="step_synthesis",
                agent_id="synthesis",
                instruction="Synthesize risk and policy findings.",
                depends_on=["step_credit_risk", "step_compliance_policy"],
                expected_output="Final risk review.",
                data_source_categories=["specialist_outputs"],
            ),
        ],
    )


def _conditional_data_quality_plan() -> ExecutionPlan:
    return _plan(
        plan_id="plan_conditional_data_quality",
        selected_agents=["internal_knowledge", "data_quality", "synthesis"],
        steps=[
            PlanStep(
                step_id="step_internal_knowledge",
                agent_id="internal_knowledge",
                instruction="Review internal CRM notes.",
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
    )


def _generic_conditional_plan() -> ExecutionPlan:
    return _plan(
        plan_id="plan_generic_conditional_review",
        selected_agents=["internal_knowledge", "credit_risk"],
        steps=[
            PlanStep(
                step_id="step_internal_knowledge",
                agent_id="internal_knowledge",
                instruction="Review internal CRM notes.",
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
    )


def _generic_conditional_fan_in_plan() -> ExecutionPlan:
    return _plan(
        plan_id="plan_generic_conditional_fan_in",
        selected_agents=["internal_knowledge", "industry_research", "credit_risk"],
        steps=[
            PlanStep(
                step_id="step_internal_knowledge",
                agent_id="internal_knowledge",
                instruction="Review internal CRM notes.",
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
    )


def _colliding_route_conditional_fan_in_plan() -> ExecutionPlan:
    return _plan(
        plan_id="plan_colliding_route_conditional_fan_in",
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
                instruction="Review internal CRM notes.",
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
                instruction="Assess credit risk when review is needed.",
                depends_on=["step_internal_knowledge"],
                condition="needs-review",
                expected_output="Credit risk themes.",
                data_source_categories=["credit_risk"],
            ),
            PlanStep(
                step_id="step_compliance_policy",
                agent_id="compliance_policy",
                instruction="Review policy when review is needed.",
                depends_on=["step_internal_knowledge"],
                condition="needs_review",
                expected_output="Policy caveats.",
                data_source_categories=["compliance_policy"],
            ),
            PlanStep(
                step_id="step_synthesis",
                agent_id="synthesis",
                instruction="Synthesize selected review path and industry context.",
                depends_on=[
                    "step_credit_risk",
                    "step_compliance_policy",
                    "step_industry_research",
                ],
                expected_output="Final RM-ready review.",
                data_source_categories=["specialist_outputs"],
            ),
        ],
    )


def _mixed_conditional_fan_in_plan() -> ExecutionPlan:
    return _plan(
        plan_id="plan_mixed_conditional_fan_in",
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
                instruction="Review internal CRM notes.",
                expected_output="Internal customer context.",
                data_source_categories=["internal_crm"],
            ),
            PlanStep(
                step_id="step_industry_research",
                agent_id="industry_research",
                instruction="Gather industry context in parallel.",
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
                instruction="Synthesize internal, data quality, and industry context.",
                depends_on=["step_data_quality", "step_industry_research"],
                expected_output="Final briefing or clarification need.",
                data_source_categories=["specialist_outputs"],
            ),
        ],
    )


def _mandatory_data_quality_plan() -> ExecutionPlan:
    return _plan(
        plan_id="plan_mandatory_data_quality",
        selected_agents=["internal_knowledge", "data_quality", "synthesis"],
        steps=[
            PlanStep(
                step_id="step_internal_knowledge",
                agent_id="internal_knowledge",
                instruction="Review internal CRM notes.",
                expected_output="Internal customer context.",
                data_source_categories=["internal_crm"],
            ),
            PlanStep(
                step_id="step_data_quality",
                agent_id="data_quality",
                instruction="Always validate internal CRM data quality.",
                depends_on=["step_internal_knowledge"],
                expected_output="Data quality gaps.",
                data_source_categories=["data_quality"],
            ),
            PlanStep(
                step_id="step_synthesis",
                agent_id="synthesis",
                instruction="Synthesize validated internal context.",
                depends_on=["step_data_quality"],
                expected_output="Final briefing.",
                data_source_categories=["specialist_outputs"],
            ),
        ],
    )


def _runtime_edges(result) -> set[tuple[str, str, str | None]]:
    return set(result.runtime.edge_routes)


def test_graph_builder_rejects_unapproved_draft_plan_record() -> None:
    # Arrange
    plan = _direct_plan()
    store = ApprovalStateStore(
        agent_descriptors=[_descriptor(agent_id) for agent_id in plan.selected_agents]
    )
    draft_record = store.add_draft(plan)
    builder = GraphBuilder(registry=_registry_for(plan))

    # Act / Assert
    with pytest.raises(GraphPlanApprovalError, match="approved immutable plan"):
        builder.build(draft_record)


def test_graph_builder_rejects_approved_plan_that_is_not_immutable() -> None:
    # Arrange
    plan = _plan(
        plan_id="plan_mutable_after_approval",
        selected_agents=["internal_knowledge"],
        immutable_after_approval=False,
        steps=[
            PlanStep(
                step_id="step_internal_knowledge",
                agent_id="internal_knowledge",
                instruction="Summarize internal notes.",
                expected_output="Internal customer context.",
                data_source_categories=["internal_crm"],
            )
        ],
    )
    record = _approved_record_without_store_validation(plan)
    builder = GraphBuilder(registry=_registry_for(plan))

    # Act / Assert
    with pytest.raises(GraphPlanApprovalError, match="immutable_after_approval"):
        builder.build(record)


def test_graph_builder_builds_direct_node_graph() -> None:
    # Arrange
    plan = _direct_plan()
    record = _approval_record_for(plan)
    builder = GraphBuilder(registry=_registry_for(plan))

    # Act
    result = builder.build(record)

    # Assert
    assert result.spec.pattern == "direct"
    assert [step.agent_id for step in result.spec.steps] == ["internal_knowledge"]
    assert result.spec.edges == []
    assert result.runtime.is_adk_backed is True
    assert result.runtime.workflow.__class__.__name__ == "Workflow"
    assert _runtime_edges(result) == {
        ("__START__", "graph_step_internal_knowledge", None)
    }


def test_graph_builder_builds_sequential_graph() -> None:
    # Arrange
    plan = _sequential_plan()
    registry = _registry_for(plan)
    builder = GraphBuilder(registry=registry)

    # Act
    result = builder.build(_approval_record_for(plan))

    # Assert
    assert registry.checked_plans == [plan.plan_id]
    assert result.spec.pattern == "sequential"
    assert [(edge.from_step_id, edge.to_step_id) for edge in result.spec.edges] == [
        ("graph_step_internal_knowledge", "graph_step_credit_risk"),
        ("graph_step_credit_risk", "graph_step_synthesis"),
    ]
    assert _runtime_edges(result) == {
        ("__START__", "graph_step_internal_knowledge", None),
        ("graph_step_internal_knowledge", "graph_step_credit_risk", None),
        ("graph_step_credit_risk", "graph_step_synthesis", None),
    }


def test_graph_builder_builds_fan_out_fan_in_graph_with_synthesis_join() -> None:
    # Arrange
    plan = _fan_out_fan_in_plan()
    builder = GraphBuilder(registry=_registry_for(plan))

    # Act
    result = builder.build(_approval_record_for(plan))

    # Assert
    assert result.spec.pattern == "fan_out_fan_in"
    assert result.spec.steps[-1].agent_id == "synthesis"
    assert result.spec.steps[-1].depends_on == [
        "graph_step_relationship_summary",
        "graph_step_internal_knowledge",
        "graph_step_industry_research",
    ]
    assert "join_graph_step_synthesis" in result.runtime.node_names
    assert _runtime_edges(result) == {
        ("__START__", "graph_step_relationship_summary", None),
        ("__START__", "graph_step_internal_knowledge", None),
        ("__START__", "graph_step_industry_research", None),
        ("graph_step_relationship_summary", "join_graph_step_synthesis", None),
        ("graph_step_internal_knowledge", "join_graph_step_synthesis", None),
        ("graph_step_industry_research", "join_graph_step_synthesis", None),
        ("join_graph_step_synthesis", "graph_step_synthesis", None),
    }


def test_graph_builder_builds_mixed_sequential_parallel_graph() -> None:
    # Arrange
    plan = _mixed_plan()
    builder = GraphBuilder(registry=_registry_for(plan))

    # Act
    result = builder.build(_approval_record_for(plan))

    # Assert
    assert result.spec.pattern == "mixed"
    assert "join_graph_step_synthesis" in result.runtime.node_names
    assert _runtime_edges(result) == {
        ("__START__", "graph_step_internal_knowledge", None),
        ("graph_step_internal_knowledge", "graph_step_credit_risk", None),
        ("graph_step_internal_knowledge", "graph_step_compliance_policy", None),
        ("graph_step_credit_risk", "join_graph_step_synthesis", None),
        ("graph_step_compliance_policy", "join_graph_step_synthesis", None),
        ("join_graph_step_synthesis", "graph_step_synthesis", None),
    }


def test_graph_builder_builds_conditional_data_quality_branch() -> None:
    # Arrange
    plan = _conditional_data_quality_plan()
    builder = GraphBuilder(registry=_registry_for(plan))

    # Act
    result = builder.build(_approval_record_for(plan))

    # Assert
    assert result.spec.pattern == "conditional"
    assert {
        (edge.from_step_id, edge.to_step_id, edge.condition)
        for edge in result.spec.edges
    } == {
        (
            "graph_step_internal_knowledge",
            "graph_step_data_quality",
            "missing_internal_data",
        ),
        ("graph_step_data_quality", "graph_step_synthesis", None),
        ("graph_step_internal_knowledge", "graph_step_synthesis", "__DEFAULT__"),
    }
    assert _runtime_edges(result) == {
        ("__START__", "graph_step_internal_knowledge", None),
        (
            "graph_step_internal_knowledge",
            "graph_step_data_quality",
            "missing_internal_data",
        ),
        ("graph_step_data_quality", "graph_step_synthesis", None),
        ("graph_step_internal_knowledge", "graph_step_synthesis", "__DEFAULT__"),
    }


def test_graph_builder_preserves_mandatory_data_quality_dependency() -> None:
    # Arrange
    plan = _mandatory_data_quality_plan()
    builder = GraphBuilder(registry=_registry_for(plan))

    # Act
    result = builder.build(_approval_record_for(plan))

    # Assert
    assert result.spec.pattern == "sequential"
    assert {
        (edge.from_step_id, edge.to_step_id, edge.condition)
        for edge in result.spec.edges
    } == {
        ("graph_step_internal_knowledge", "graph_step_data_quality", None),
        ("graph_step_data_quality", "graph_step_synthesis", None),
    }
    assert _runtime_edges(result) == {
        ("__START__", "graph_step_internal_knowledge", None),
        ("graph_step_internal_knowledge", "graph_step_data_quality", None),
        ("graph_step_data_quality", "graph_step_synthesis", None),
    }


def test_graph_builder_preserves_generic_plan_step_conditions() -> None:
    # Arrange
    plan = _generic_conditional_plan()
    builder = GraphBuilder(registry=_registry_for(plan))

    # Act
    result = builder.build(_approval_record_for(plan))

    # Assert
    assert result.spec.pattern == "conditional"
    assert {
        (edge.from_step_id, edge.to_step_id, edge.condition)
        for edge in result.spec.edges
    } == {
        (
            "graph_step_internal_knowledge",
            "graph_step_credit_risk",
            "needs_review",
        )
    }
    assert _runtime_edges(result) == {
        ("__START__", "graph_step_internal_knowledge", None),
        (
            "graph_step_internal_knowledge",
            "graph_step_credit_risk",
            "needs_review",
        ),
    }


def test_graph_builder_joins_generic_conditional_fan_in_dependencies() -> None:
    # Arrange
    plan = _generic_conditional_fan_in_plan()
    builder = GraphBuilder(registry=_registry_for(plan))

    # Act
    result = builder.build(_approval_record_for(plan))

    # Assert
    assert result.spec.pattern == "conditional"
    assert {
        (edge.from_step_id, edge.to_step_id, edge.condition)
        for edge in result.spec.edges
    } == {
        (
            "graph_step_internal_knowledge",
            "graph_step_credit_risk",
            "needs_review",
        ),
        ("graph_step_industry_research", "graph_step_credit_risk", None),
    }
    assert "join_graph_step_credit_risk" in result.runtime.node_names
    assert "join_graph_step_credit_risk_needs_review_gate" in (
        result.runtime.node_names
    )
    assert _runtime_edges(result) == {
        ("__START__", "graph_step_internal_knowledge", None),
        ("__START__", "graph_step_industry_research", None),
        (
            "graph_step_internal_knowledge",
            "join_graph_step_credit_risk_needs_review_gate",
            "needs_review",
        ),
        (
            "join_graph_step_credit_risk_needs_review_gate",
            "join_graph_step_credit_risk",
            None,
        ),
        (
            "graph_step_industry_research",
            "join_graph_step_credit_risk",
            None,
        ),
        ("join_graph_step_credit_risk", "graph_step_credit_risk", None),
    }


def test_graph_builder_uses_unique_join_names_for_colliding_route_suffixes() -> None:
    # Arrange
    plan = _colliding_route_conditional_fan_in_plan()
    builder = GraphBuilder(registry=_registry_for(plan))

    # Act
    result = builder.build(_approval_record_for(plan))

    # Assert
    synthesis_join_names = [
        name
        for name in result.runtime.node_names
        if name.startswith("join_graph_step_synthesis_needs_review")
    ]
    assert synthesis_join_names == [
        "join_graph_step_synthesis_needs_review_route1",
        "join_graph_step_synthesis_needs_review_route2",
    ]
    assert _runtime_edges(result) == {
        ("__START__", "graph_step_internal_knowledge", None),
        ("__START__", "graph_step_industry_research", None),
        (
            "graph_step_internal_knowledge",
            "graph_step_credit_risk",
            "needs-review",
        ),
        (
            "graph_step_internal_knowledge",
            "graph_step_compliance_policy",
            "needs_review",
        ),
        (
            "graph_step_credit_risk",
            "join_graph_step_synthesis_needs_review_route1",
            None,
        ),
        (
            "graph_step_industry_research",
            "join_graph_step_synthesis_needs_review_route1",
            None,
        ),
        (
            "join_graph_step_synthesis_needs_review_route1",
            "graph_step_synthesis",
            None,
        ),
        (
            "graph_step_compliance_policy",
            "join_graph_step_synthesis_needs_review_route2",
            None,
        ),
        (
            "graph_step_industry_research",
            "join_graph_step_synthesis_needs_review_route2",
            None,
        ),
        (
            "join_graph_step_synthesis_needs_review_route2",
            "graph_step_synthesis",
            None,
        ),
    }


def test_graph_builder_joins_mixed_conditional_fan_in_branches() -> None:
    # Arrange
    plan = _mixed_conditional_fan_in_plan()
    builder = GraphBuilder(registry=_registry_for(plan))

    # Act
    result = builder.build(_approval_record_for(plan))

    # Assert
    assert result.spec.pattern == "conditional"
    assert {
        (edge.from_step_id, edge.to_step_id, edge.condition)
        for edge in result.spec.edges
    } == {
        (
            "graph_step_internal_knowledge",
            "graph_step_data_quality",
            "missing_internal_data",
        ),
        ("graph_step_data_quality", "graph_step_synthesis", None),
        ("graph_step_industry_research", "graph_step_synthesis", None),
        ("graph_step_internal_knowledge", "graph_step_synthesis", "__DEFAULT__"),
    }
    assert (
        "join_graph_step_synthesis_missing_internal_data" in result.runtime.node_names
    )
    assert (
        "join_graph_step_synthesis_default_gate" in result.runtime.node_names
    )
    assert "join_graph_step_synthesis_default" in result.runtime.node_names
    assert _runtime_edges(result) == {
        ("__START__", "graph_step_internal_knowledge", None),
        ("__START__", "graph_step_industry_research", None),
        (
            "graph_step_internal_knowledge",
            "graph_step_data_quality",
            "missing_internal_data",
        ),
        (
            "graph_step_data_quality",
            "join_graph_step_synthesis_missing_internal_data",
            None,
        ),
        (
            "graph_step_industry_research",
            "join_graph_step_synthesis_missing_internal_data",
            None,
        ),
        (
            "graph_step_internal_knowledge",
            "join_graph_step_synthesis_default_gate",
            "__DEFAULT__",
        ),
        (
            "join_graph_step_synthesis_default_gate",
            "join_graph_step_synthesis_default",
            None,
        ),
        (
            "graph_step_industry_research",
            "join_graph_step_synthesis_default",
            None,
        ),
        (
            "join_graph_step_synthesis_missing_internal_data",
            "graph_step_synthesis",
            None,
        ),
        ("join_graph_step_synthesis_default", "graph_step_synthesis", None),
    }


def test_graph_builder_validates_approved_plan_agents_are_available() -> None:
    # Arrange
    plan = _fan_out_fan_in_plan()
    record = _approval_record_for(plan)
    registry = _Registry(
        [
            "relationship_summary",
            "internal_knowledge",
            "synthesis",
        ]
    )
    builder = GraphBuilder(registry=registry)

    # Act / Assert
    with pytest.raises(UnavailableAgentError, match="industry_research"):
        builder.build(record)
