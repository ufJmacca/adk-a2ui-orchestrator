"""Application service wiring classification, approval, execution, and A2UI routing."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from orchestrator_demo.a2a_support.transport import DataPart
from orchestrator_demo.a2ui_support.approval_canvas import build_approval_canvas
from orchestrator_demo.a2ui_support.renderer_contract import (
    prepare_approval_a2ui_for_renderer,
    prepare_specialist_a2ui_for_renderer,
)
from orchestrator_demo.agents import SpecialistAgent, build_default_specialists
from orchestrator_demo.app.logging import log_audit_event
from orchestrator_demo.contracts import (
    AgentDescriptor,
    ExecutionPlan,
    RoutingDecision,
    SpecialistRequest,
    SpecialistResponse,
    StatusEvent,
)
from orchestrator_demo.intent.classifier import (
    DeterministicIntentClassifier,
    IntentClassifier,
)
from orchestrator_demo.intent.slm_mock_client import MockSlmIntentClient, SlmIntentClient
from orchestrator_demo.orchestrator.approval_state import (
    ApprovalActionResult,
    ApprovalRecord,
    ApprovalStateStore,
)
from orchestrator_demo.orchestrator.graph_runtime import (
    AdkGraphRuntime,
    GraphExecutionResult,
    GraphRuntimeError,
)
from orchestrator_demo.orchestrator.planner import (
    DraftExecutionPlanner,
    PlanCreationError,
)
from orchestrator_demo.orchestrator.request_context import (
    RequestContext,
    SpecialistPreApprovalError,
    call_specialist_with_guard,
)
from orchestrator_demo.orchestrator.router import RequestRouter
from orchestrator_demo.orchestrator.surface_routes import (
    SurfaceOwner,
    SurfaceRouteRegistry,
    SurfaceRouteResult,
)
from orchestrator_demo.registry.agent_registry import AgentRegistry


@dataclass(frozen=True)
class OrchestratorRequestResult:
    """Result returned for a new natural-language user request."""

    path: str
    decision: RoutingDecision
    context: RequestContext
    specialist_responses: tuple[SpecialistResponse, ...] = ()
    a2ui_parts: tuple[DataPart, ...] = ()
    approval_plan: ExecutionPlan | None = None
    approval_result: ApprovalActionResult | None = None
    graph_execution: GraphExecutionResult | None = None
    status_events: tuple[StatusEvent, ...] = ()
    final_artifacts: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OrchestratorUserActionResult:
    """Result returned after a structured A2UI userAction event."""

    status: str
    approval_result: ApprovalActionResult | None = None
    surface_route_result: SurfaceRouteResult | None = None
    specialist_responses: tuple[SpecialistResponse, ...] = ()
    a2ui_parts: tuple[DataPart, ...] = ()
    graph_execution: GraphExecutionResult | None = None
    status_events: tuple[StatusEvent, ...] = ()
    final_artifacts: dict[str, Any] = field(default_factory=dict)


class OrchestratorService:
    """High-level service for the local ADK/A2UI orchestrator demo."""

    def __init__(
        self,
        *,
        registry: AgentRegistry | None = None,
        slm_client: SlmIntentClient | None = None,
        intent_classifier: IntentClassifier | None = None,
        specialists: Mapping[str, SpecialistAgent] | None = None,
        specialist_user_action_adapters: Mapping[str, Any] | None = None,
        surface_registry: SurfaceRouteRegistry | None = None,
    ) -> None:
        self._registry = registry or AgentRegistry.from_default_config()
        self._specialists = dict(
            build_default_specialists() if specialists is None else specialists
        )
        self._specialist_user_action_adapters = _default_user_action_adapters(
            self._specialists
        )
        self._specialist_user_action_adapters.update(
            specialist_user_action_adapters or {}
        )
        self._surface_registry = surface_registry or SurfaceRouteRegistry()
        self._contexts_by_plan_id: dict[str, RequestContext] = {}

        self._router = RequestRouter(
            slm_client=slm_client or MockSlmIntentClient(),
            intent_classifier=intent_classifier or DeterministicIntentClassifier(),
            registry=self._registry,
        )
        self._planner = DraftExecutionPlanner(registry=self._registry)
        self._approval_store = ApprovalStateStore(
            agent_descriptors=self._registry.descriptors(),
            graph_runtime=_GuardedGraphRuntime(
                specialists=self._specialists,
                contexts_by_plan_id=self._contexts_by_plan_id,
            ),
        )

    async def handle_user_request(self, user_input: str) -> OrchestratorRequestResult:
        """Classify a user request and either call one specialist or propose a plan."""

        context = await self._router.route_request(user_input)
        if context.decision.path == "direct":
            return await self._handle_direct_request(context)
        if context.decision.path == "plan_required":
            return self._handle_plan_required_request(context)

        return OrchestratorRequestResult(
            path=context.decision.path,
            decision=context.decision,
            context=context,
        )

    async def handle_user_action(self, user_action: Any) -> OrchestratorUserActionResult:
        """Route a structured A2UI event to approval state or specialist owner."""

        route_result = await self._surface_registry.route_user_action(
            user_action,
            specialist_adapters=self._specialist_user_action_adapters,
        )
        if route_result.status == "orchestrator_owned":
            return self._handle_orchestrator_owned_user_action(user_action)

        specialist_response = _coerce_specialist_response(route_result.response)
        owner_agent_id = (
            route_result.owner.owner_id
            if route_result.owner is not None
            and route_result.owner.owner_type == "specialist"
            else None
        )
        a2ui_parts = self._prepare_specialist_response_a2ui(
            specialist_response,
            owner_agent_id=owner_agent_id,
        )
        return OrchestratorUserActionResult(
            status=route_result.status,
            surface_route_result=route_result,
            specialist_responses=(
                (specialist_response,) if specialist_response is not None else ()
            ),
            a2ui_parts=a2ui_parts,
            final_artifacts=_final_artifacts_for_response(specialist_response),
        )

    def specialist_call_counts(self) -> dict[str, int]:
        """Return non-zero specialist call counts keyed by agent id."""

        return {
            agent_id: specialist.call_count
            for agent_id, specialist in self._specialists.items()
            if specialist.call_count
        }

    def surface_owner(self, surface_id: str) -> SurfaceOwner | None:
        """Return the deterministic owner registered for an A2UI surface."""

        return self._surface_registry.owner_for(surface_id)

    def approval_record(self, plan_id: str) -> ApprovalRecord:
        """Return the stored approval record snapshot for a plan."""

        return self._approval_store.get(plan_id)

    async def _handle_direct_request(
        self,
        context: RequestContext,
    ) -> OrchestratorRequestResult:
        selected_agent = context.decision.selected_agent
        if selected_agent is None:
            raise SpecialistPreApprovalError("direct route is missing selected_agent")

        specialist = self._specialists.get(selected_agent)
        if specialist is None:
            return _direct_route_handler_unavailable_result(context, selected_agent)

        request = SpecialistRequest(
            request_id=(
                f"request_direct_{_contract_id_token(selected_agent)}_"
                f"{_contract_id_token(context.plan_scope_id)}"
            ),
            user_input=context.user_input,
            agent_id=selected_agent,
        )
        response = await call_specialist_with_guard(
            context,
            request,
            specialist.handle,
        )
        a2ui_parts = self._prepare_specialist_response_a2ui(
            response,
            owner_agent_id=selected_agent,
        )

        return OrchestratorRequestResult(
            path="direct",
            decision=context.decision,
            context=context,
            specialist_responses=(response,),
            a2ui_parts=a2ui_parts,
            final_artifacts=_final_artifacts_for_response(response),
        )

    def _handle_plan_required_request(
        self,
        context: RequestContext,
    ) -> OrchestratorRequestResult:
        try:
            plan = self._planner.create_plan(context, record_draft=False)
        except PlanCreationError as exc:
            return _plan_creation_failed_result(context, exc)

        missing_handler_agent_ids = _missing_handler_agent_ids(
            plan,
            self._specialists,
        )
        if missing_handler_agent_ids:
            return _plan_route_handler_unavailable_result(
                context,
                missing_handler_agent_ids,
            )

        context.record_draft_plan(plan)
        self._approval_store.add_draft(plan)
        self._contexts_by_plan_id[plan.plan_id] = context
        log_audit_event(
            "plan_proposed",
            {
                "plan_id": plan.plan_id,
                "approval_surface_id": plan.approval_surface_id,
                "detected_intents": list(plan.detected_intents),
                "selected_agent_ids": list(plan.selected_agents),
                "step_ids": [step.step_id for step in plan.steps],
                "step_count": len(plan.steps),
                "approval_required": True,
            },
        )
        a2ui_parts = prepare_approval_a2ui_for_renderer(
            build_approval_canvas(
                plan,
                agent_descriptors=_executable_agent_descriptors(
                    self._registry.descriptors(),
                    self._specialists,
                ),
            ),
            plan_id=plan.plan_id,
            surface_registry=self._surface_registry,
        )

        return OrchestratorRequestResult(
            path="plan_required",
            decision=context.decision,
            context=context,
            approval_plan=plan,
            a2ui_parts=tuple(a2ui_parts),
        )

    def _handle_orchestrator_owned_user_action(
        self,
        user_action: Any,
    ) -> OrchestratorUserActionResult:
        self._approval_store.replace_agent_descriptors(
            _executable_agent_descriptors(
                self._registry.descriptors(),
                self._specialists,
            )
        )
        approval_result = self._approval_store.apply_user_action(user_action)
        a2ui_parts = self._prepare_approval_result_a2ui(approval_result)
        graph_execution = approval_result.graph_execution
        specialist_responses = (
            graph_execution.specialist_responses
            if graph_execution is not None
            else ()
        )
        specialist_a2ui_parts = self._prepare_graph_response_a2ui(graph_execution)

        return OrchestratorUserActionResult(
            status=approval_result.status,
            approval_result=approval_result,
            specialist_responses=specialist_responses,
            a2ui_parts=(*a2ui_parts, *specialist_a2ui_parts),
            graph_execution=graph_execution,
            status_events=(
                graph_execution.status_events if graph_execution is not None else ()
            ),
            final_artifacts=_final_artifacts_for_graph(graph_execution),
        )

    def _prepare_approval_result_a2ui(
        self,
        approval_result: ApprovalActionResult,
    ) -> tuple[DataPart, ...]:
        if approval_result.status == "draft_updated":
            assert approval_result.plan_id is not None
            self._sync_context_draft(approval_result.plan_id)
            return tuple(
                prepare_approval_a2ui_for_renderer(
                    approval_result.refreshed_a2ui_parts,
                    plan_id=approval_result.plan_id,
                    surface_registry=self._surface_registry,
                )
            )

        return ()

    def _prepare_graph_response_a2ui(
        self,
        graph_execution: GraphExecutionResult | None,
    ) -> tuple[DataPart, ...]:
        if graph_execution is None:
            return ()

        parts: list[DataPart] = []
        for owned_response in graph_execution.owned_specialist_responses:
            parts.extend(
                self._prepare_specialist_response_a2ui(
                    owned_response.response,
                    owner_agent_id=owned_response.owner_agent_id,
                )
            )
        return tuple(parts)

    def _prepare_specialist_response_a2ui(
        self,
        response: SpecialistResponse | None,
        *,
        owner_agent_id: str | None,
    ) -> tuple[DataPart, ...]:
        if response is None or response.a2ui_payload is None:
            return ()
        if owner_agent_id is None:
            raise SpecialistPreApprovalError(
                "specialist A2UI ownership requires a known invoked agent"
            )

        return tuple(
            prepare_specialist_a2ui_for_renderer(
                response.a2ui_payload,
                owner_agent_id=owner_agent_id,
                surface_registry=self._surface_registry,
            )
        )

    def _sync_context_draft(self, plan_id: str) -> None:
        context = self._contexts_by_plan_id.get(plan_id)
        if context is None:
            return

        record = self._approval_store.get(plan_id)
        context.record_draft_plan(record.draft_plan)


class _GuardedGraphRuntime:
    """Graph runtime wrapper that enforces request approval guardrails."""

    def __init__(
        self,
        *,
        specialists: Mapping[str, SpecialistAgent],
        contexts_by_plan_id: Mapping[str, RequestContext],
    ) -> None:
        self._specialists = dict(specialists)
        self._contexts_by_plan_id = contexts_by_plan_id

    def execute(self, plan: ExecutionPlan) -> GraphExecutionResult:
        context = self._contexts_by_plan_id.get(plan.plan_id)
        if context is None:
            raise GraphRuntimeError(
                f"no routed request context is registered for approved plan {plan.plan_id}"
            )
        had_structured_approval = context.has_structured_approval
        context.mark_plan_approved(plan)
        runtime = AdkGraphRuntime(specialist_handlers=self._guarded_handlers())
        try:
            return runtime.execute(plan)
        except Exception:
            if not had_structured_approval:
                context.rollback_plan_approval(plan)
            raise

    def _guarded_handlers(self) -> dict[str, Any]:
        return {
            agent_id: self._guarded_handler(agent_id, specialist)
            for agent_id, specialist in self._specialists.items()
        }

    def _guarded_handler(self, agent_id: str, specialist: SpecialistAgent) -> Any:
        async def handle(request: SpecialistRequest) -> SpecialistResponse:
            if request.plan_id is None:
                raise SpecialistPreApprovalError(
                    "graph specialist calls must include an approved plan_id"
                )
            context = self._contexts_by_plan_id.get(request.plan_id)
            if context is None:
                raise SpecialistPreApprovalError(
                    "graph specialist call targets an unknown approved plan"
                )
            if request.agent_id != agent_id:
                raise SpecialistPreApprovalError(
                    f"handler {agent_id!r} cannot execute request for {request.agent_id!r}"
                )
            return await call_specialist_with_guard(context, request, specialist.handle)

        return handle


class _LocalSpecialistUserActionAdapter:
    """Default local handler for specialist-owned A2UI userAction events."""

    def __init__(self, *, agent_id: str) -> None:
        self._agent_id = agent_id

    async def handle_user_action(self, _user_action: Any) -> SpecialistResponse:
        return SpecialistResponse(
            response_id=f"response_{_contract_id_token(self._agent_id)}_user_action",
            agent_id=self._agent_id,
            content=(
                f"{self._agent_id.replace('_', ' ').title()} Agent: "
                "A2UI user action handled."
            ),
            structured_output={"status": "handled", "agent_id": self._agent_id},
        )


def _default_user_action_adapters(
    specialists: Mapping[str, SpecialistAgent],
) -> dict[str, Any]:
    adapters: dict[str, Any] = {}
    for agent_id, specialist in specialists.items():
        if callable(getattr(specialist, "handle_user_action", None)):
            adapters[agent_id] = specialist
        else:
            adapters[agent_id] = _LocalSpecialistUserActionAdapter(agent_id=agent_id)
    return adapters


def _coerce_specialist_response(value: Any) -> SpecialistResponse | None:
    if value is None:
        return None
    if isinstance(value, SpecialistResponse):
        return value
    if isinstance(value, Mapping):
        try:
            return SpecialistResponse.model_validate(value)
        except ValueError:
            return None
    return None


def _final_artifacts_for_response(
    response: SpecialistResponse | None,
) -> dict[str, Any]:
    if response is None:
        return {}
    return {"final_response": response}


def _direct_route_handler_unavailable_result(
    context: RequestContext,
    selected_agent: str,
) -> OrchestratorRequestResult:
    decision = RoutingDecision(
        path="clarification_required",
        selected_agent=None,
        confidence=context.decision.confidence,
        reason=(
            "A safe direct route cannot be executed because no specialist "
            f"handler is registered for selected agent {selected_agent}."
        ),
    )
    context.decision = decision
    log_audit_event(
        "route_decision",
        {
            "path": decision.path,
            "selected_agent": decision.selected_agent,
            "confidence": decision.confidence,
            "reason": decision.reason,
            "missing_handler_agent_id": selected_agent,
        },
    )
    return OrchestratorRequestResult(
        path=decision.path,
        decision=decision,
        context=context,
        final_artifacts={
            "error": {
                "code": "specialist_handler_unavailable",
                "agent_id": selected_agent,
                "message": decision.reason,
            }
        },
    )


def _plan_route_handler_unavailable_result(
    context: RequestContext,
    agent_ids: list[str],
) -> OrchestratorRequestResult:
    unavailable = ", ".join(agent_ids)
    decision = RoutingDecision(
        path="clarification_required",
        selected_agent=None,
        confidence=context.decision.confidence,
        reason=(
            "A safe approval plan cannot be created because no specialist "
            f"handler is registered for planned agents: {unavailable}."
        ),
    )
    context.decision = decision
    log_audit_event(
        "route_decision",
        {
            "path": decision.path,
            "selected_agent": decision.selected_agent,
            "confidence": decision.confidence,
            "reason": decision.reason,
            "missing_handler_agent_ids": agent_ids,
        },
    )
    return OrchestratorRequestResult(
        path=decision.path,
        decision=decision,
        context=context,
        final_artifacts={
            "error": {
                "code": "specialist_handler_unavailable",
                "agent_ids": agent_ids,
                "message": decision.reason,
            }
        },
    )


def _plan_creation_failed_result(
    context: RequestContext,
    exc: PlanCreationError,
) -> OrchestratorRequestResult:
    decision = RoutingDecision(
        path="clarification_required",
        selected_agent=None,
        confidence=context.decision.confidence,
        reason=f"A safe approval plan cannot be created. {exc}",
    )
    context.decision = decision
    log_audit_event(
        "route_decision",
        {
            "path": decision.path,
            "selected_agent": decision.selected_agent,
            "confidence": decision.confidence,
            "reason": decision.reason,
            "planner_error": str(exc),
        },
    )
    return OrchestratorRequestResult(
        path=decision.path,
        decision=decision,
        context=context,
        final_artifacts={
            "error": {
                "code": "plan_creation_failed",
                "message": decision.reason,
            }
        },
    )


def _executable_agent_descriptors(
    descriptors: Sequence[AgentDescriptor],
    specialists: Mapping[str, SpecialistAgent],
) -> list[AgentDescriptor]:
    return [
        descriptor
        for descriptor in descriptors
        if descriptor.agent_id in specialists
    ]


def _missing_handler_agent_ids(
    plan: ExecutionPlan,
    specialists: Mapping[str, SpecialistAgent],
) -> list[str]:
    missing: list[str] = []
    for agent_id in (*plan.selected_agents, *(step.agent_id for step in plan.steps)):
        if agent_id in specialists or agent_id in missing:
            continue
        missing.append(agent_id)
    return missing


def _final_artifacts_for_graph(
    graph_execution: GraphExecutionResult | None,
) -> dict[str, Any]:
    if graph_execution is None:
        return {}

    final_response = (
        graph_execution.specialist_responses[-1]
        if graph_execution.specialist_responses
        else None
    )
    return {
        "final_response": final_response,
        "specialist_responses": graph_execution.specialist_responses,
        "status_events": graph_execution.status_events,
    }


def _contract_id_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    token = token.strip("_-")
    return token if token and token[0].isalnum() else "generated"


__all__ = [
    "OrchestratorRequestResult",
    "OrchestratorService",
    "OrchestratorUserActionResult",
]
