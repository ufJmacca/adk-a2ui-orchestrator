from __future__ import annotations

import logging

import pytest

from orchestrator_demo.contracts import ExecutionPlan, PlanStep, SpecialistResponse
from orchestrator_demo.orchestrator.graph_runtime import (
    AdkGraphRuntime,
    GraphRuntimeError,
)


AUDIT_LOGGER_NAME = "orchestrator_demo.audit"


def _response_for(request) -> SpecialistResponse:
    return SpecialistResponse(
        response_id=f"response_{request.request_id.removeprefix('request_')}",
        agent_id=request.agent_id,
        content=f"{request.agent_id} completed {request.step_id}",
        structured_output={"agent_id": request.agent_id, "step_id": request.step_id},
    )


def _fan_out_fan_in_plan() -> ExecutionPlan:
    return ExecutionPlan(
        plan_id="plan_progress_parallel",
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
        approval_surface_id="surface_plan_progress_parallel",
    )


def _missing_agent_plan() -> ExecutionPlan:
    return ExecutionPlan(
        plan_id="plan_progress_missing_agent",
        objective="Run an approved plan with an unavailable specialist.",
        detected_intents=["meeting_prep"],
        selected_agents=["internal_knowledge", "unavailable_agent"],
        steps=[
            PlanStep(
                step_id="step_internal_knowledge",
                agent_id="internal_knowledge",
                instruction="Review internal CRM notes.",
                expected_output="Internal context.",
            ),
            PlanStep(
                step_id="step_missing_agent",
                agent_id="unavailable_agent",
                instruction="Run a specialist that is no longer registered.",
                depends_on=["step_internal_knowledge"],
                expected_output="Unavailable specialist output.",
            ),
        ],
        approval_surface_id="surface_plan_progress_missing_agent",
    )


def _handler_failure_plan() -> ExecutionPlan:
    return ExecutionPlan(
        plan_id="plan_progress_handler_failure",
        objective="Run an approved plan where one specialist fails.",
        detected_intents=["credit_risk"],
        selected_agents=["internal_knowledge", "credit_risk"],
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
            ),
        ],
        approval_surface_id="surface_plan_progress_handler_failure",
    )


def test_progress_events_include_required_statuses_in_harness_visible_order() -> None:
    # Arrange
    plan = _fan_out_fan_in_plan()
    runtime = AdkGraphRuntime(
        specialist_handlers={
            agent_id: _response_for for agent_id in plan.selected_agents
        }
    )

    # Act
    result = runtime.execute(plan)

    # Assert
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
    assert result.status_events[0].details == {"planVersion": 1}
    assert [
        event.details["parallelGroup"]
        for event in result.status_events
        if event.status == "parallel_branch_started"
    ] == [
        "parallel_meeting_context",
        "parallel_meeting_context",
        "parallel_meeting_context",
    ]
    assert result.status_events[-1].details == {"responseCount": 4}


def test_unavailable_approved_plan_agent_fails_with_developer_status_event(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    plan = _missing_agent_plan()
    called_requests = []

    def internal_handler(request):
        called_requests.append(request)
        return _response_for(request)

    runtime = AdkGraphRuntime(
        specialist_handlers={"internal_knowledge": internal_handler}
    )

    # Act / Assert
    with caplog.at_level(logging.INFO, logger=AUDIT_LOGGER_NAME):
        with pytest.raises(GraphRuntimeError, match="unavailable_agent") as exc_info:
            runtime.execute(plan)

    failure = exc_info.value
    assert called_requests == []
    assert [event.status for event in failure.status_events] == [
        "plan_approved",
        "graph_created",
        "step_failed",
    ]
    failed_event = failure.status_events[-1]
    assert failed_event.step_id == "graph_step_missing_agent"
    assert failed_event.message == (
        "Approved plan step step_missing_agent failed before execution: "
        "no specialist handler registered for agent unavailable_agent."
    )
    assert failed_event.details == {
        "agentId": "unavailable_agent",
        "planStepId": "step_missing_agent",
        "developerMessage": (
            "Register agent unavailable_agent before executing approved plan "
            "plan_progress_missing_agent."
        ),
    }
    failed_audit_records = [
        record
        for record in caplog.records
        if getattr(record, "audit_event", None) == "graph_execution_failed"
    ]
    assert len(failed_audit_records) == 1
    assert getattr(failed_audit_records[0], "event_payload") == {
        "graph_id": "graph_progress_missing_agent",
        "plan_id": "plan_progress_missing_agent",
        "error_type": "GraphRuntimeError",
        "status_event_count": 3,
        "request_count": 0,
        "response_count": 0,
    }


def test_registered_specialist_failure_preserves_prior_events_and_step_failed_status() -> None:
    # Arrange
    plan = _handler_failure_plan()

    def failing_credit_handler(request):
        raise RuntimeError("credit service timed out")

    runtime = AdkGraphRuntime(
        specialist_handlers={
            "internal_knowledge": _response_for,
            "credit_risk": failing_credit_handler,
        }
    )

    # Act / Assert
    with pytest.raises(GraphRuntimeError, match="credit_risk") as exc_info:
        runtime.execute(plan)

    failure = exc_info.value
    assert [event.status for event in failure.status_events] == [
        "plan_approved",
        "graph_created",
        "step_started",
        "step_completed",
        "step_started",
        "step_failed",
    ]
    assert [request.step_id for request in failure.specialist_requests] == [
        "step_internal_knowledge",
        "step_credit_risk",
    ]
    assert [response.agent_id for response in failure.specialist_responses] == [
        "internal_knowledge"
    ]
    failed_event = failure.status_events[-1]
    assert failed_event.step_id == "graph_step_credit_risk"
    assert failed_event.message == (
        "Approved plan step step_credit_risk failed during execution: "
        "RuntimeError: credit service timed out."
    )
    assert failed_event.details == {
        "agentId": "credit_risk",
        "planStepId": "step_credit_risk",
        "errorType": "RuntimeError",
        "developerMessage": (
            "Specialist handler for agent credit_risk raised RuntimeError while "
            "executing approved plan plan_progress_handler_failure step "
            "step_credit_risk."
        ),
    }


def test_registered_specialist_failure_redacts_secret_like_exception_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    plan = _handler_failure_plan()
    leaked_value = "OPENROUTER_API_KEY=sk-live-handler-secret-token-123456789"

    def failing_credit_handler(request):
        raise RuntimeError(f"provider rejected {leaked_value}")

    runtime = AdkGraphRuntime(
        specialist_handlers={
            "internal_knowledge": _response_for,
            "credit_risk": failing_credit_handler,
        }
    )

    # Act / Assert
    with pytest.raises(GraphRuntimeError, match="credit_risk") as exc_info:
        runtime.execute(plan)

    failure = exc_info.value
    failed_event = failure.status_events[-1]
    exposed_failure = repr(
        {
            "runtime_error": str(failure),
            "event_message": failed_event.message,
            "event_details": failed_event.details,
        }
    )
    assert leaked_value not in exposed_failure
    assert "sk-live-handler-secret-token-123456789" not in exposed_failure
    assert leaked_value not in caplog.text
    assert "sk-live-handler-secret-token-123456789" not in caplog.text
    assert "<redacted-secret>" in exposed_failure
    assert failed_event.message == (
        "Approved plan step step_credit_risk failed during execution: "
        "RuntimeError: provider rejected <redacted-secret>."
    )
