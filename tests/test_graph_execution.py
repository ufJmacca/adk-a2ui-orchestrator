from __future__ import annotations

from collections.abc import Callable

from orchestrator_demo.contracts import ExecutionPlan, PlanStep, SpecialistResponse
from orchestrator_demo.orchestrator.graph_runtime import AdkGraphRuntime


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
