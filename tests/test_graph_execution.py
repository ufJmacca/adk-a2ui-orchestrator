from __future__ import annotations

import asyncio
from collections.abc import Callable

from orchestrator_demo.contracts import (
    ExecutionPlan,
    PlanStep,
    SpecialistRequest,
    SpecialistResponse,
)
from orchestrator_demo.orchestrator.graph_runtime import (
    AdkGraphRuntime,
    default_specialist_handlers,
)


def _response_for(request) -> SpecialistResponse:
    dependency_outputs = request.context.get("dependencyOutputs", {})
    return SpecialistResponse(
        response_id=f"response_{request.request_id.removeprefix('request_')}",
        agent_id=request.agent_id,
        content=f"{request.agent_id} completed {request.step_id}",
        structured_output={
            "agent_id": request.agent_id,
            "step_id": request.step_id,
            "dependency_step_ids": list(dependency_outputs),
        },
    )


def _recording_handlers(
    agent_ids: list[str],
    observed: list[tuple[str, str | None]],
) -> dict[str, Callable]:
    def handler(request):
        observed.append((request.agent_id, request.step_id))
        return _response_for(request)

    return {agent_id: handler for agent_id in agent_ids}


def test_default_specialist_handlers_use_wrappers_for_remote_compatible_agents() -> None:
    # Arrange
    from orchestrator_demo.a2a_support.local_remote_wrapper import (
        LocalRemoteAgentWrapper,
        REDACTED_SECRET_VALUE,
    )

    handlers = default_specialist_handlers(
        ["credit_risk", "internal_knowledge", "product_opportunity"]
    )
    secret_value = "sk-" + "testsecret1234567890"

    # Act / Assert
    for agent_id in ("internal_knowledge", "product_opportunity"):
        handler = handlers[agent_id]
        wrapper = getattr(handler, "__self__", None)
        assert isinstance(wrapper, LocalRemoteAgentWrapper)

        request = SpecialistRequest(
            request_id=f"request_{agent_id}_secret_redaction",
            user_input=f"Summarize context with OPENROUTER_API_KEY={secret_value}",
            agent_id=agent_id,
            context={
                "customer": "ABC Manufacturing",
                "api_key": secret_value,
            },
        )
        response = asyncio.run(handler(request))

        assert response.agent_id == agent_id
        assert wrapper.calls[-1].user_input == REDACTED_SECRET_VALUE
        assert wrapper.calls[-1].context["api_key"] == REDACTED_SECRET_VALUE


def _sequential_plan() -> ExecutionPlan:
    return ExecutionPlan(
        plan_id="plan_execution_sequential",
        objective="Prepare a sequential risk review.",
        detected_intents=["meeting_prep"],
        selected_agents=["internal_knowledge", "credit_risk", "synthesis"],
        steps=[
            PlanStep(
                step_id="step_internal_knowledge",
                agent_id="internal_knowledge",
                instruction="Review internal CRM notes.",
                expected_output="Internal context.",
            ),
            PlanStep(
                step_id="step_credit_risk",
                agent_id="credit_risk",
                instruction="Assess credit themes.",
                depends_on=["step_internal_knowledge"],
                expected_output="Credit risk themes.",
            ),
            PlanStep(
                step_id="step_synthesis",
                agent_id="synthesis",
                instruction="Synthesize the risk review.",
                depends_on=["step_credit_risk"],
                expected_output="Final RM-ready risk review.",
            ),
        ],
        approval_surface_id="surface_plan_execution_sequential",
    )


def _fan_out_fan_in_plan() -> ExecutionPlan:
    return ExecutionPlan(
        plan_id="plan_execution_parallel",
        objective="Prepare a meeting briefing from parallel workstreams.",
        detected_intents=["meeting_prep"],
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
                expected_output="Relationship context.",
                parallel_group="parallel_meeting_context",
            ),
            PlanStep(
                step_id="step_internal_knowledge",
                agent_id="internal_knowledge",
                instruction="Review internal CRM notes.",
                expected_output="Internal context.",
                parallel_group="parallel_meeting_context",
            ),
            PlanStep(
                step_id="step_industry_research",
                agent_id="industry_research",
                instruction="Assess industry risks.",
                expected_output="Industry context.",
                parallel_group="parallel_meeting_context",
            ),
            PlanStep(
                step_id="step_synthesis",
                agent_id="synthesis",
                instruction="Synthesize the meeting brief.",
                depends_on=[
                    "step_relationship_summary",
                    "step_internal_knowledge",
                    "step_industry_research",
                ],
                expected_output="Final RM-ready meeting brief.",
            ),
        ],
        approval_surface_id="surface_plan_execution_parallel",
    )


def _hyphenated_step_id_plan() -> ExecutionPlan:
    return ExecutionPlan(
        plan_id="plan_execution_hyphenated_ids",
        objective="Prepare a risk review with custom step IDs.",
        detected_intents=["credit_risk"],
        selected_agents=["credit_risk", "compliance_policy", "synthesis"],
        steps=[
            PlanStep(
                step_id="step_credit-risk",
                agent_id="credit_risk",
                instruction="Assess credit risk themes.",
                expected_output="Credit risk themes.",
            ),
            PlanStep(
                step_id="step_credit_risk",
                agent_id="compliance_policy",
                instruction="Assess policy constraints.",
                expected_output="Policy constraints.",
            ),
            PlanStep(
                step_id="step_synthesis",
                agent_id="synthesis",
                instruction="Synthesize the risk and policy review.",
                depends_on=["step_credit-risk", "step_credit_risk"],
                expected_output="Final RM-ready review.",
            ),
        ],
        approval_surface_id="surface_plan_execution_hyphenated_ids",
    )


def _mixed_sequential_parallel_plan() -> ExecutionPlan:
    return ExecutionPlan(
        plan_id="plan_execution_mixed_parallel",
        objective="Prepare a mixed risk and policy review.",
        detected_intents=["credit_risk"],
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
                expected_output="Internal context.",
            ),
            PlanStep(
                step_id="step_credit_risk",
                agent_id="credit_risk",
                instruction="Assess credit risk themes.",
                depends_on=["step_internal_knowledge"],
                expected_output="Credit risk themes.",
                parallel_group="parallel_risk_policy_review",
            ),
            PlanStep(
                step_id="step_compliance_policy",
                agent_id="compliance_policy",
                instruction="Assess policy constraints.",
                depends_on=["step_internal_knowledge"],
                expected_output="Policy constraints.",
                parallel_group="parallel_risk_policy_review",
            ),
            PlanStep(
                step_id="step_synthesis",
                agent_id="synthesis",
                instruction="Synthesize the risk and policy review.",
                depends_on=["step_credit_risk", "step_compliance_policy"],
                expected_output="Final RM-ready review.",
            ),
        ],
        approval_surface_id="surface_plan_execution_mixed_parallel",
    )


def _conditional_data_quality_plan() -> ExecutionPlan:
    return ExecutionPlan(
        plan_id="plan_execution_conditional_data_quality",
        objective="Prepare a briefing and check data quality only if needed.",
        detected_intents=["meeting_prep"],
        selected_agents=["internal_knowledge", "data_quality", "synthesis"],
        steps=[
            PlanStep(
                step_id="step_internal_knowledge",
                agent_id="internal_knowledge",
                instruction="Review internal CRM notes.",
                expected_output="Internal context.",
            ),
            PlanStep(
                step_id="step_data_quality",
                agent_id="data_quality",
                instruction="Check internal data gaps.",
                depends_on=["step_internal_knowledge"],
                expected_output="Data quality gaps.",
            ),
            PlanStep(
                step_id="step_synthesis",
                agent_id="synthesis",
                instruction="Synthesize the available context.",
                depends_on=["step_data_quality"],
                expected_output="Final RM-ready brief.",
            ),
        ],
        approval_surface_id="surface_plan_execution_conditional_data_quality",
    )


def _mixed_conditional_fan_in_plan() -> ExecutionPlan:
    return ExecutionPlan(
        plan_id="plan_execution_mixed_conditional_fan_in",
        objective="Prepare a briefing from public and internal context.",
        detected_intents=["meeting_prep"],
        selected_agents=["internal_knowledge", "web_search", "data_quality", "synthesis"],
        steps=[
            PlanStep(
                step_id="step_internal_knowledge",
                agent_id="internal_knowledge",
                instruction="Review internal CRM notes.",
                expected_output="Internal context.",
            ),
            PlanStep(
                step_id="step_web_search",
                agent_id="web_search",
                instruction="Gather public context.",
                expected_output="Public context.",
            ),
            PlanStep(
                step_id="step_data_quality",
                agent_id="data_quality",
                instruction="Check internal data gaps.",
                depends_on=["step_internal_knowledge"],
                expected_output="Data quality gaps.",
            ),
            PlanStep(
                step_id="step_synthesis",
                agent_id="synthesis",
                instruction="Synthesize all available context.",
                depends_on=["step_data_quality", "step_web_search"],
                expected_output="Final RM-ready brief.",
            ),
        ],
        approval_surface_id="surface_plan_execution_mixed_conditional_fan_in",
    )


def test_sequential_workflow_executes_steps_in_plan_order_and_collects_outputs() -> None:
    # Arrange
    plan = _sequential_plan()
    observed: list[tuple[str, str | None]] = []
    runtime = AdkGraphRuntime(
        specialist_handlers=_recording_handlers(plan.selected_agents, observed)
    )

    # Act
    result = runtime.execute(plan)

    # Assert
    assert observed == [
        ("internal_knowledge", "step_internal_knowledge"),
        ("credit_risk", "step_credit_risk"),
        ("synthesis", "step_synthesis"),
    ]
    assert [request.step_id for request in result.specialist_requests] == [
        "step_internal_knowledge",
        "step_credit_risk",
        "step_synthesis",
    ]
    assert [response.agent_id for response in result.specialist_responses] == [
        "internal_knowledge",
        "credit_risk",
        "synthesis",
    ]
    synthesis_request = result.specialist_requests[-1]
    assert synthesis_request.context["dependencyOutputs"] == {
        "step_credit_risk": {
            "response_id": "response_plan_execution_sequential_step_credit_risk",
            "agent_id": "credit_risk",
            "content": "credit_risk completed step_credit_risk",
            "structured_output": {
                "agent_id": "credit_risk",
                "step_id": "step_credit_risk",
                "dependency_step_ids": ["step_internal_knowledge"],
            },
            "a2ui_payload": None,
            "surface_id": None,
        }
    }


def test_fan_out_fan_in_workflow_executes_deterministically_and_synthesizes_outputs() -> None:
    # Arrange
    plan = _fan_out_fan_in_plan()
    observed: list[tuple[str, str | None]] = []
    runtime = AdkGraphRuntime(
        specialist_handlers=_recording_handlers(plan.selected_agents, observed)
    )

    # Act
    result = runtime.execute(plan)

    # Assert
    assert result.graph.pattern == "fan_out_fan_in"
    assert observed == [
        ("relationship_summary", "step_relationship_summary"),
        ("internal_knowledge", "step_internal_knowledge"),
        ("industry_research", "step_industry_research"),
        ("synthesis", "step_synthesis"),
    ]
    synthesis_request = result.specialist_requests[-1]
    assert synthesis_request.agent_id == "synthesis"
    assert list(synthesis_request.context["dependencyOutputs"]) == [
        "step_relationship_summary",
        "step_internal_knowledge",
        "step_industry_research",
    ]
    assert result.specialist_responses[-1].structured_output == {
        "agent_id": "synthesis",
        "step_id": "step_synthesis",
        "dependency_step_ids": [
            "step_relationship_summary",
            "step_internal_knowledge",
            "step_industry_research",
        ],
    }


def test_hyphenated_step_ids_create_valid_adk_node_names_and_preserve_plan_ids() -> None:
    # Arrange
    plan = _hyphenated_step_id_plan()
    observed: list[tuple[str, str | None]] = []
    runtime = AdkGraphRuntime(
        specialist_handlers=_recording_handlers(plan.selected_agents, observed)
    )

    # Act
    result = runtime.execute(plan)

    # Assert
    graph_step_ids = [step.graph_step_id for step in result.graph.steps]
    assert graph_step_ids == [
        "graph_step_credit_risk",
        "graph_step_credit_risk_2",
        "graph_step_synthesis",
    ]
    assert all(graph_step_id.isidentifier() for graph_step_id in graph_step_ids)
    assert [request.step_id for request in result.specialist_requests] == [
        "step_credit-risk",
        "step_credit_risk",
        "step_synthesis",
    ]
    synthesis_request = result.specialist_requests[-1]
    assert list(synthesis_request.context["dependencyOutputs"]) == [
        "step_credit-risk",
        "step_credit_risk",
    ]


def test_mixed_sequential_parallel_workflow_reports_mixed_pattern() -> None:
    # Arrange
    plan = _mixed_sequential_parallel_plan()
    observed: list[tuple[str, str | None]] = []
    runtime = AdkGraphRuntime(
        specialist_handlers=_recording_handlers(plan.selected_agents, observed)
    )

    # Act
    result = runtime.execute(plan)

    # Assert
    assert result.graph.pattern == "mixed"
    assert observed == [
        ("internal_knowledge", "step_internal_knowledge"),
        ("credit_risk", "step_credit_risk"),
        ("compliance_policy", "step_compliance_policy"),
        ("synthesis", "step_synthesis"),
    ]


def test_conditional_data_quality_workflow_preserves_default_branch() -> None:
    # Arrange
    plan = _conditional_data_quality_plan()
    observed: list[tuple[str, str | None]] = []
    runtime = AdkGraphRuntime(
        specialist_handlers=_recording_handlers(plan.selected_agents, observed)
    )

    # Act
    result = runtime.execute(plan)

    # Assert
    assert result.graph.pattern == "conditional"
    assert {
        (edge.from_step_id, edge.to_step_id, edge.condition)
        for edge in result.graph.edges
    } == {
        (
            "graph_step_internal_knowledge",
            "graph_step_data_quality",
            "missing_internal_data",
        ),
        ("graph_step_data_quality", "graph_step_synthesis", None),
        ("graph_step_internal_knowledge", "graph_step_synthesis", "__DEFAULT__"),
    }
    assert observed == [
        ("internal_knowledge", "step_internal_knowledge"),
        ("synthesis", "step_synthesis"),
    ]
    synthesis_request = result.specialist_requests[-1]
    assert list(synthesis_request.context["dependencyOutputs"]) == [
        "step_internal_knowledge"
    ]
    assert result.specialist_responses[-1].structured_output["dependency_step_ids"] == [
        "step_internal_knowledge"
    ]


def test_conditional_data_quality_workflow_can_take_missing_data_branch() -> None:
    # Arrange
    plan = _conditional_data_quality_plan()
    observed: list[tuple[str, str | None]] = []
    handlers = _recording_handlers(plan.selected_agents, observed)

    def internal_handler(request) -> SpecialistResponse:
        observed.append((request.agent_id, request.step_id))
        response = _response_for(request)
        response.structured_output["graphRoute"] = "missing_internal_data"
        return response

    handlers["internal_knowledge"] = internal_handler
    runtime = AdkGraphRuntime(specialist_handlers=handlers)

    # Act
    result = runtime.execute(plan)

    # Assert
    assert observed == [
        ("internal_knowledge", "step_internal_knowledge"),
        ("data_quality", "step_data_quality"),
        ("synthesis", "step_synthesis"),
    ]
    synthesis_request = result.specialist_requests[-1]
    assert list(synthesis_request.context["dependencyOutputs"]) == [
        "step_data_quality"
    ]


def test_mixed_conditional_fan_in_default_branch_runs_synthesis_once() -> None:
    # Arrange
    plan = _mixed_conditional_fan_in_plan()
    observed: list[tuple[str, str | None]] = []
    runtime = AdkGraphRuntime(
        specialist_handlers=_recording_handlers(plan.selected_agents, observed)
    )

    # Act
    result = runtime.execute(plan)

    # Assert
    assert observed == [
        ("internal_knowledge", "step_internal_knowledge"),
        ("web_search", "step_web_search"),
        ("synthesis", "step_synthesis"),
    ]
    assert [agent_id for agent_id, _ in observed].count("synthesis") == 1
    synthesis_request = result.specialist_requests[-1]
    assert synthesis_request.agent_id == "synthesis"
    assert list(synthesis_request.context["dependencyOutputs"]) == [
        "step_web_search",
        "step_internal_knowledge",
    ]


def test_mixed_conditional_fan_in_missing_data_branch_runs_synthesis_once() -> None:
    # Arrange
    plan = _mixed_conditional_fan_in_plan()
    observed: list[tuple[str, str | None]] = []
    handlers = _recording_handlers(plan.selected_agents, observed)

    def internal_handler(request) -> SpecialistResponse:
        observed.append((request.agent_id, request.step_id))
        response = _response_for(request)
        response.structured_output["graphRoute"] = "missing_internal_data"
        return response

    handlers["internal_knowledge"] = internal_handler
    runtime = AdkGraphRuntime(specialist_handlers=handlers)

    # Act
    result = runtime.execute(plan)

    # Assert
    assert observed == [
        ("internal_knowledge", "step_internal_knowledge"),
        ("web_search", "step_web_search"),
        ("data_quality", "step_data_quality"),
        ("synthesis", "step_synthesis"),
    ]
    assert [agent_id for agent_id, _ in observed].count("synthesis") == 1
    synthesis_request = result.specialist_requests[-1]
    assert synthesis_request.agent_id == "synthesis"
    assert list(synthesis_request.context["dependencyOutputs"]) == [
        "step_data_quality",
        "step_web_search",
    ]
