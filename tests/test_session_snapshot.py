from __future__ import annotations

import json
from typing import Any

import pytest

from orchestrator_demo.a2ui_support.renderer_contract import (
    prepare_approval_a2ui_for_renderer,
    prepare_specialist_a2ui_for_renderer,
)
from orchestrator_demo.a2ui_support.schema_manager import A2UI_VERSION, BASIC_CATALOG_ID
from orchestrator_demo.contracts import (
    AgentDescriptor,
    ExecutionPlan,
    GraphEdge,
    GraphSpec,
    GraphStep,
    IntentSuggestion,
    LlmIntentAssessment,
    PlanStep,
    RoutingDecision,
    SpecialistRequest,
    SpecialistResponse,
    StatusEvent,
)
from orchestrator_demo.orchestrator.approval_state import ApprovalStateStore
from orchestrator_demo.orchestrator.graph_runtime import GraphExecutionResult
from orchestrator_demo.orchestrator.request_context import (
    RequestContext,
    SpecialistPreApprovalError,
)
from orchestrator_demo.orchestrator.surface_routes import SurfaceRouteRegistry


KNOWN_SECRET = "OPENROUTER_API_KEY=sk-or-v1-session-secret-should-not-appear"


def _descriptor(agent_id: str) -> AgentDescriptor:
    return AgentDescriptor(
        agent_id=agent_id,
        display_name=agent_id.replace("_", " ").title(),
        capabilities=[agent_id.replace("_", " ")],
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        a2ui_catalogs=["basic"],
        routing_examples=[],
        execution_mode="local_llm",
    )


def _agent_descriptors() -> list[AgentDescriptor]:
    return [
        _descriptor("relationship_summary"),
        _descriptor("internal_knowledge"),
        _descriptor("industry_research"),
        _descriptor("credit_risk"),
        _descriptor("synthesis"),
    ]


def _plan(
    plan_id: str,
    *,
    objective: str = "Prepare a meeting briefing for ABC Manufacturing.",
    plan_version: int = 1,
) -> ExecutionPlan:
    return ExecutionPlan(
        plan_id=plan_id,
        objective=objective,
        detected_intents=["meeting_prep", "internal_knowledge"],
        selected_agents=["internal_knowledge", "synthesis"],
        steps=[
            PlanStep(
                step_id="step_internal_knowledge",
                agent_id="internal_knowledge",
                instruction=f"Review internal context for: {objective}",
                expected_output="Internal customer context.",
                data_source_categories=["internal_crm"],
                parallel_group="parallel_context",
            ),
            PlanStep(
                step_id="step_synthesis",
                agent_id="synthesis",
                instruction=f"Create the RM-ready brief for: {objective}",
                depends_on=["step_internal_knowledge"],
                expected_output="Final meeting brief.",
                data_source_categories=["specialist_outputs"],
            ),
        ],
        data_source_categories=["internal_crm"],
        risk_notes=["Synthetic demo data only."],
        approval_surface_id=f"surface_{plan_id}",
        plan_version=plan_version,
    )


def _approval_action(plan: ExecutionPlan) -> dict[str, Any]:
    return {
        "userAction": {
            "type": "approve_plan",
            "surfaceId": plan.approval_surface_id,
            "payload": {
                "planId": plan.plan_id,
                "editedPlanVersion": plan.plan_version,
                "approvedStepIds": [step.step_id for step in plan.steps],
            },
        }
    }


def _rejection_action(plan: ExecutionPlan) -> dict[str, Any]:
    return {
        "userAction": {
            "type": "reject_plan",
            "surfaceId": plan.approval_surface_id,
            "payload": {
                "planId": plan.plan_id,
                "editedPlanVersion": plan.plan_version,
                "reason": "The requested scope should be narrowed first.",
            },
        }
    }


def _graph_for(plan: ExecutionPlan) -> GraphSpec:
    return GraphSpec(
        graph_id=f"graph_{plan.plan_id.removeprefix('plan_')}",
        plan_id=plan.plan_id,
        pattern="sequential",
        steps=[
            GraphStep(
                graph_step_id=f"graph_step_{step.step_id.removeprefix('step_')}",
                plan_step_id=step.step_id,
                agent_id=step.agent_id,
                depends_on=[
                    f"graph_step_{dependency.removeprefix('step_')}"
                    for dependency in step.depends_on
                ],
                parallel_group=step.parallel_group,
            )
            for step in plan.steps
        ],
        edges=[
            GraphEdge(
                from_step_id=f"graph_step_{dependency.removeprefix('step_')}",
                to_step_id=f"graph_step_{step.step_id.removeprefix('step_')}",
            )
            for step in plan.steps
            for dependency in step.depends_on
        ],
    )


class SuccessfulGraphRuntime:
    def execute(self, plan: ExecutionPlan) -> GraphExecutionResult:
        request = SpecialistRequest(
            request_id=f"request_{plan.plan_id.removeprefix('plan_')}_internal",
            user_input=plan.steps[0].instruction,
            agent_id=plan.steps[0].agent_id,
            plan_id=plan.plan_id,
            step_id=plan.steps[0].step_id,
            context={"objective": plan.objective},
        )
        response = SpecialistResponse(
            response_id=f"response_{plan.plan_id.removeprefix('plan_')}_internal",
            agent_id=plan.steps[0].agent_id,
            content="Internal Knowledge Agent: completed.",
        )
        return GraphExecutionResult(
            graph=_graph_for(plan),
            workflow=object(),
            status_events=(
                StatusEvent(
                    event_id=f"event_{plan.plan_id.removeprefix('plan_')}_done",
                    graph_id=f"graph_{plan.plan_id.removeprefix('plan_')}",
                    plan_id=plan.plan_id,
                    status="final_response_ready",
                    message="Final response ready.",
                ),
            ),
            specialist_requests=(request,),
            specialist_responses=(response,),
            specialist_response_requests=(request,),
            adk_event_outputs=(),
        )


class FailingGraphRuntime:
    def execute(self, plan: ExecutionPlan) -> GraphExecutionResult:
        raise RuntimeError(f"graph failed for {plan.plan_id}")


class RecordingSpecialistAdapter:
    def __init__(self) -> None:
        self.received_user_actions: list[Any] = []

    async def handle_user_action(self, user_action: Any) -> dict[str, str]:
        self.received_user_actions.append(user_action)
        return {"status": "handled"}


def _context_for_plan(plan: ExecutionPlan) -> RequestContext:
    context = RequestContext(
        user_input=plan.objective,
        slm_suggestion=IntentSuggestion(intent="meeting_prep", confidence=0.82),
        llm_assessment=LlmIntentAssessment(
            intents=["meeting_prep", "internal_knowledge"],
            confidence=0.91,
            complexity="complex",
            rationale="Meeting prep needs approval before specialist execution.",
            required_agents=["internal_knowledge", "synthesis"],
        ),
        decision=RoutingDecision(
            path="plan_required",
            selected_agent=None,
            confidence=0.874,
            reason="Plan approval required.",
        ),
        plan_scope_id=plan.plan_id.removeprefix("plan_snapshot_"),
    )
    context.record_draft_plan(plan)
    return context


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


def _specialist_a2ui(
    surface_id: str,
    *,
    text: str = "Treasury services fit the stated need.",
) -> list[dict[str, Any]]:
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
                        "text": text,
                    }
                ],
            },
        },
    ]


def test_session_snapshot_exports_json_safe_schema_and_redacts_secret_like_state() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.approval_canvas import build_approval_canvas
    from orchestrator_demo.a2ui_support.secret_safety import REDACTED_SECRET
    from orchestrator_demo.orchestrator.session_snapshot import (
        SNAPSHOT_SCHEMA_VERSION,
        export_session_snapshot,
    )

    store = ApprovalStateStore(
        agent_descriptors=_agent_descriptors(),
        graph_runtime=SuccessfulGraphRuntime(),
    )
    draft_plan = _plan("plan_draft")
    sensitive_draft_plan = _plan(
        "plan_sensitive_draft",
        objective=f"Prepare a meeting briefing without persisting {KNOWN_SECRET}",
    )
    approved_plan = _plan("plan_approved")
    rejected_plan = _plan("plan_rejected")
    store.add_draft(draft_plan)
    store.add_draft(sensitive_draft_plan)
    store.add_draft(approved_plan)
    store.add_draft(rejected_plan)
    store.apply_user_action(_approval_action(approved_plan))
    store.apply_user_action(_rejection_action(rejected_plan))

    failed_store = ApprovalStateStore(
        agent_descriptors=_agent_descriptors(),
        graph_runtime=FailingGraphRuntime(),
    )
    failed_plan = _plan("plan_failed")
    failed_store.add_draft(failed_plan)
    with pytest.raises(RuntimeError, match="graph failed for plan_failed"):
        failed_store.apply_user_action(_approval_action(failed_plan))
    store.restore_records(
        {
            **store.export_records(),
            **failed_store.export_records(),
        }
    )

    surface_registry = SurfaceRouteRegistry()
    prepare_approval_a2ui_for_renderer(
        build_approval_canvas(draft_plan, agent_descriptors=_agent_descriptors()),
        plan_id=draft_plan.plan_id,
        surface_registry=surface_registry,
    )
    context_plan = _plan(
        "plan_snapshot_sensitive_scope",
        objective=f"Prepare guarded context without persisting {KNOWN_SECRET}",
    )
    request_context = _context_for_plan(context_plan)
    artifact_refs = {
        "orchestrator_latest_result.json": {
            "uri": "file://.adk/artifacts/orchestrator_latest_result.json",
            "diagnostic": KNOWN_SECRET,
        }
    }
    assert KNOWN_SECRET in sensitive_draft_plan.objective
    assert KNOWN_SECRET in request_context.user_input

    # Act
    snapshot = export_session_snapshot(
        approval_store=store,
        request_contexts_by_plan_id={context_plan.plan_id: request_context},
        surface_registry=surface_registry,
        artifact_refs=artifact_refs,
    )
    serialized = json.dumps(snapshot, sort_keys=True)

    # Assert
    assert snapshot["schemaVersion"] == SNAPSHOT_SCHEMA_VERSION == 1
    assert set(snapshot) == {
        "schemaVersion",
        "approvalRecords",
        "requestContextsByPlanId",
        "surfaceRegistry",
        "artifactRefs",
    }
    assert set(snapshot["approvalRecords"]) == {
        "plan_draft",
        "plan_sensitive_draft",
        "plan_approved",
        "plan_rejected",
        "plan_failed",
    }
    assert {
        plan_id: record["status"]
        for plan_id, record in snapshot["approvalRecords"].items()
    } == {
        "plan_draft": "draft",
        "plan_sensitive_draft": "draft",
        "plan_approved": "approved",
        "plan_rejected": "rejected",
        "plan_failed": "approved_execution_failed",
    }
    assert KNOWN_SECRET not in serialized
    assert REDACTED_SECRET in json.dumps(
        snapshot["approvalRecords"]["plan_sensitive_draft"],
        sort_keys=True,
    )
    assert REDACTED_SECRET in json.dumps(
        snapshot["requestContextsByPlanId"][context_plan.plan_id],
        sort_keys=True,
    )
    assert REDACTED_SECRET in json.dumps(
        snapshot["artifactRefs"]["orchestrator_latest_result.json"],
        sort_keys=True,
    )


def test_approval_records_round_trip_all_plan_states() -> None:
    # Arrange
    store = ApprovalStateStore(
        agent_descriptors=_agent_descriptors(),
        graph_runtime=SuccessfulGraphRuntime(),
    )
    draft_plan = _plan("plan_draft")
    approved_plan = _plan("plan_approved")
    rejected_plan = _plan("plan_rejected")
    store.add_draft(draft_plan)
    store.add_draft(approved_plan)
    store.add_draft(rejected_plan)
    store.apply_user_action(_approval_action(approved_plan))
    store.apply_user_action(_rejection_action(rejected_plan))

    failed_store = ApprovalStateStore(
        agent_descriptors=_agent_descriptors(),
        graph_runtime=FailingGraphRuntime(),
    )
    failed_plan = _plan("plan_failed")
    failed_store.add_draft(failed_plan)
    with pytest.raises(RuntimeError):
        failed_store.apply_user_action(_approval_action(failed_plan))
    exported_records = {
        **store.export_records(),
        **failed_store.export_records(),
    }
    restored = ApprovalStateStore(
        agent_descriptors=_agent_descriptors(),
        graph_runtime=SuccessfulGraphRuntime(),
    )

    # Act
    restored.restore_records(exported_records)

    # Assert
    assert restored.get("plan_draft").status == "draft"
    approved_record = restored.get("plan_approved")
    assert approved_record.status == "approved"
    assert approved_record.approved_plan == approved_plan
    assert approved_record.approved_version == approved_plan.plan_version
    rejected_record = restored.get("plan_rejected")
    assert rejected_record.status == "rejected"
    assert rejected_record.rejection_reason == (
        "The requested scope should be narrowed first."
    )
    failed_record = restored.get("plan_failed")
    assert failed_record.status == "approved_execution_failed"
    assert failed_record.approved_plan == failed_plan
    assert failed_record.execution_failure_reason == (
        "RuntimeError: graph failed for plan_failed"
    )
    assert json.loads(json.dumps(exported_records)) == exported_records


@pytest.mark.asyncio
async def test_request_context_round_trip_preserves_approval_guardrails() -> None:
    # Arrange
    plan = _plan(
        "plan_snapshot_guard_scope",
        objective="Prepare a guarded meeting briefing.",
    )
    context = _context_for_plan(plan)
    approved_step = plan.steps[0]
    context.mark_plan_approved(plan)
    snapshot = context.export_snapshot()
    restored = RequestContext.restore_snapshot(snapshot)
    allowed_request = SpecialistRequest(
        request_id="request_guard_allowed",
        user_input=approved_step.instruction,
        agent_id=approved_step.agent_id,
        plan_id=plan.plan_id,
        step_id=approved_step.step_id,
        context=_approved_context_for_step(plan, approved_step),
    )
    tampered_request = allowed_request.model_copy(
        update={"user_input": "Review a different customer."}
    )

    # Act
    restored.require_specialist_call_allowed(allowed_request)

    # Assert
    with pytest.raises(SpecialistPreApprovalError, match="approved instruction"):
        restored.require_specialist_call_allowed(tampered_request)
    assert restored.approved_plan_id == plan.plan_id
    assert restored.approved_plan_step_agents[plan.plan_id] == {
        "step_internal_knowledge": "internal_knowledge",
        "step_synthesis": "synthesis",
    }
    assert json.loads(json.dumps(snapshot)) == snapshot


@pytest.mark.asyncio
async def test_surface_registry_round_trip_preserves_owners_and_component_graphs() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.approval_canvas import build_approval_canvas

    plan = _plan("plan_surface")
    registry = SurfaceRouteRegistry()
    prepare_approval_a2ui_for_renderer(
        build_approval_canvas(plan, agent_descriptors=_agent_descriptors()),
        plan_id=plan.plan_id,
        surface_registry=registry,
    )
    prepare_specialist_a2ui_for_renderer(
        _specialist_a2ui("surface_product_recommendation"),
        owner_agent_id="product_opportunity",
        surface_registry=registry,
    )
    exported = registry.export_snapshot()
    restored = SurfaceRouteRegistry()
    adapter = RecordingSpecialistAdapter()
    user_action = {
        "userAction": {
            "type": "specialist_action",
            "surfaceId": "surface_product_recommendation",
            "payload": {"buttonId": "show_more_detail"},
        }
    }

    # Act
    restored.restore_snapshot(exported)
    result = await restored.route_user_action(
        user_action,
        specialist_adapters={"product_opportunity": adapter},
    )

    # Assert
    assert restored.export_snapshot() == exported
    assert exported["ownersBySurfaceId"]["surface_plan_surface"]["ownerType"] == (
        "orchestrator"
    )
    assert exported["ownersBySurfaceId"]["surface_product_recommendation"][
        "ownerType"
    ] == "specialist"
    assert "surface_plan_surface" in exported["componentsBySurfaceId"]
    assert (
        exported["componentsBySurfaceId"]["surface_product_recommendation"]["root"][
            "text"
        ]
        == "Treasury services fit the stated need."
    )
    assert result.status == "forwarded"
    assert adapter.received_user_actions == [user_action]


def test_session_snapshot_restore_rehydrates_all_state_groups() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.approval_canvas import build_approval_canvas
    from orchestrator_demo.orchestrator.session_snapshot import (
        export_session_snapshot,
        restore_session_snapshot,
    )

    plan = _plan(
        "plan_snapshot_restore_scope",
        objective="Prepare a restorable meeting briefing.",
    )
    context = _context_for_plan(plan)
    approval_store = ApprovalStateStore(
        agent_descriptors=_agent_descriptors(),
        graph_runtime=SuccessfulGraphRuntime(),
    )
    approval_store.add_draft(plan)
    surface_registry = SurfaceRouteRegistry()
    prepare_approval_a2ui_for_renderer(
        build_approval_canvas(plan, agent_descriptors=_agent_descriptors()),
        plan_id=plan.plan_id,
        surface_registry=surface_registry,
    )
    artifact_refs = {
        "orchestrator_plan_plan_snapshot_restore_scope_execution.json": {
            "uri": "file://.adk/artifacts/plan_snapshot_restore_scope.json"
        }
    }
    snapshot = export_session_snapshot(
        approval_store=approval_store,
        request_contexts_by_plan_id={plan.plan_id: context},
        surface_registry=surface_registry,
        artifact_refs=artifact_refs,
    )

    # Act
    restored = restore_session_snapshot(
        snapshot,
        agent_descriptors=_agent_descriptors(),
        graph_runtime=SuccessfulGraphRuntime(),
    )

    # Assert
    assert restored.approval_store.get(plan.plan_id).draft_plan == plan
    assert restored.request_contexts_by_plan_id[plan.plan_id].draft_plan_id == (
        plan.plan_id
    )
    assert restored.surface_registry.owner_for(plan.approval_surface_id).plan_id == (
        plan.plan_id
    )
    assert restored.artifact_refs == artifact_refs
