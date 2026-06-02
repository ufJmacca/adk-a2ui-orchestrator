"""Application service wiring classification, approval, execution, and A2UI routing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from orchestrator_demo.a2a_support.local_remote_wrapper import (
    build_default_local_remote_wrappers,
)
from orchestrator_demo.a2a_support.transport import A2APart, TextPart
from orchestrator_demo.a2ui_support.approval_canvas import build_approval_canvas
from orchestrator_demo.a2ui_support.renderer_contract import (
    prepare_approval_a2ui_for_renderer,
    prepare_specialist_a2ui_for_renderer,
)
from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui
from orchestrator_demo.agents import SpecialistAgent, build_default_specialists
from orchestrator_demo.app.logging import log_audit_event
from orchestrator_demo.contracts import (
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
from orchestrator_demo.orchestrator.planner import DraftExecutionPlanner
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
    a2ui_parts: tuple[A2APart, ...] = ()
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
    a2ui_parts: tuple[A2APart, ...] = ()
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
            specialists
            if specialists is not None
            else _default_specialists(self._registry)
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
            agent_descriptors=self._registry.descriptors,
            graph_runtime=_GuardedGraphRuntime(
                registry=self._registry,
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
        specialist_response, a2ui_parts = self._prepare_specialist_response_for_result(
            specialist_response,
            register_surfaces=False,
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
            decision = RoutingDecision(
                path="clarification_required",
                selected_agent=None,
                confidence=context.decision.confidence,
                reason=(
                    "The selected direct-route agent is unavailable because no "
                    f"specialist handler is registered: {selected_agent}."
                ),
            )
            context.decision = decision
            return OrchestratorRequestResult(
                path=decision.path,
                decision=decision,
                context=context,
            )

        request = SpecialistRequest(
            request_id=f"request_direct_{selected_agent}",
            user_input=context.user_input,
            agent_id=selected_agent,
        )
        response = await call_specialist_with_guard(
            context,
            request,
            specialist.handle,
        )
        safe_response, a2ui_parts = self._prepare_specialist_response_for_result(
            response,
            register_surfaces=True,
        )
        assert safe_response is not None

        return OrchestratorRequestResult(
            path="direct",
            decision=context.decision,
            context=context,
            specialist_responses=(safe_response,),
            a2ui_parts=a2ui_parts,
            final_artifacts=_final_artifacts_for_response(safe_response),
        )

    def _handle_plan_required_request(
        self,
        context: RequestContext,
    ) -> OrchestratorRequestResult:
        plan = self._planner.create_plan(context)
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
                agent_descriptors=self._registry.descriptors(),
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
        approval_result = self._approval_store.apply_user_action(user_action)
        a2ui_parts = self._prepare_approval_result_a2ui(approval_result)
        graph_execution = approval_result.graph_execution
        specialist_responses = (
            graph_execution.specialist_responses
            if graph_execution is not None
            else ()
        )
        specialist_responses, specialist_a2ui_parts = (
            self._prepare_graph_responses_for_result(specialist_responses)
        )

        return OrchestratorUserActionResult(
            status=approval_result.status,
            approval_result=approval_result,
            specialist_responses=specialist_responses,
            a2ui_parts=(*a2ui_parts, *specialist_a2ui_parts),
            graph_execution=graph_execution,
            status_events=(
                graph_execution.status_events
                if graph_execution is not None
                else approval_result.graph_status_events
            ),
            final_artifacts=_final_artifacts_for_graph(
                graph_execution,
                specialist_responses=specialist_responses,
            ),
        )

    def _prepare_approval_result_a2ui(
        self,
        approval_result: ApprovalActionResult,
    ) -> tuple[A2APart, ...]:
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

    def _prepare_graph_responses_for_result(
        self,
        responses: tuple[SpecialistResponse, ...],
    ) -> tuple[tuple[SpecialistResponse, ...], tuple[A2APart, ...]]:
        safe_responses: list[SpecialistResponse] = []
        parts: list[A2APart] = []
        for response in responses:
            safe_response, response_parts = self._prepare_specialist_response_for_result(
                response,
                register_surfaces=True,
            )
            if safe_response is not None:
                safe_responses.append(safe_response)
            parts.extend(response_parts)
        return tuple(safe_responses), tuple(parts)

    def _prepare_specialist_response_for_result(
        self,
        response: SpecialistResponse | None,
        *,
        register_surfaces: bool,
    ) -> tuple[SpecialistResponse | None, tuple[A2APart, ...]]:
        if response is None:
            return None, ()

        if register_surfaces:
            a2ui_parts = self._prepare_specialist_response_a2ui(response)
        else:
            a2ui_parts = _validated_specialist_response_a2ui(response)

        return _redact_rejected_a2ui_payload(response, a2ui_parts), a2ui_parts

    def _prepare_specialist_response_a2ui(
        self,
        response: SpecialistResponse | None,
    ) -> tuple[A2APart, ...]:
        if response is None or response.a2ui_payload is None:
            return ()

        return tuple(
            prepare_specialist_a2ui_for_renderer(
                response.a2ui_payload,
                owner_agent_id=response.agent_id,
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
        registry: AgentRegistry,
        specialists: Mapping[str, SpecialistAgent],
        contexts_by_plan_id: Mapping[str, RequestContext],
    ) -> None:
        self._registry = registry
        self._specialists = dict(specialists)
        self._contexts_by_plan_id = contexts_by_plan_id

    def execute(self, plan: ExecutionPlan) -> GraphExecutionResult:
        context = self._contexts_by_plan_id.get(plan.plan_id)
        if context is None:
            raise GraphRuntimeError(
                f"no routed request context is registered for approved plan {plan.plan_id}"
            )
        unavailable_agent_ids = set(self._registry.find_unavailable_plan_agents(plan))
        if unavailable_agent_ids:
            runtime = AdkGraphRuntime(
                specialist_handlers=self._guarded_handlers(
                    exclude_agent_ids=unavailable_agent_ids
                )
            )
            return runtime.execute(plan)
        was_approved_plan_id = context.approved_plan_id
        context.mark_plan_approved(plan)
        runtime = AdkGraphRuntime(specialist_handlers=self._guarded_handlers())
        try:
            return runtime.execute(plan)
        except Exception:
            if was_approved_plan_id is None:
                context.roll_back_failed_plan_approval(plan)
            raise

    def _guarded_handlers(
        self,
        *,
        exclude_agent_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        excluded = exclude_agent_ids or set()
        return {
            agent_id: self._guarded_handler(agent_id, specialist)
            for agent_id, specialist in self._specialists.items()
            if agent_id not in excluded
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
            response_id=f"response_{self._agent_id}_user_action",
            agent_id=self._agent_id,
            content=(
                f"{self._agent_id.replace('_', ' ').title()} Agent: "
                "A2UI user action handled."
            ),
            structured_output={"status": "handled", "agent_id": self._agent_id},
        )


def _default_specialists(registry: AgentRegistry) -> dict[str, SpecialistAgent]:
    specialists = build_default_specialists()
    wrappers = build_default_local_remote_wrappers(specialists)
    for agent_id, wrapper in wrappers.items():
        descriptor = registry.get(agent_id)
        if (
            descriptor is not None
            and descriptor.execution_mode == "local_a2a_compatible"
        ):
            specialists[agent_id] = wrapper
    return specialists


def _default_user_action_adapters(
    specialists: Mapping[str, SpecialistAgent],
) -> dict[str, Any]:
    return {
        agent_id: _default_user_action_adapter(agent_id, specialist)
        for agent_id, specialist in specialists.items()
    }


def _default_user_action_adapter(
    agent_id: str,
    specialist: SpecialistAgent,
) -> Any:
    handler = getattr(specialist, "handle_user_action", None)
    if callable(handler):
        return specialist
    return _LocalSpecialistUserActionAdapter(agent_id=agent_id)


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


def _final_artifacts_for_graph(
    graph_execution: GraphExecutionResult | None,
    *,
    specialist_responses: tuple[SpecialistResponse, ...] | None = None,
) -> dict[str, Any]:
    if graph_execution is None:
        return {}

    responses = (
        specialist_responses
        if specialist_responses is not None
        else graph_execution.specialist_responses
    )
    final_response = responses[-1] if responses else None
    return {
        "final_response": final_response,
        "specialist_responses": responses,
        "status_events": graph_execution.status_events,
    }


def _validated_specialist_response_a2ui(
    response: SpecialistResponse | None,
) -> tuple[A2APart, ...]:
    if response is None or response.a2ui_payload is None:
        return ()

    return tuple(
        validate_outbound_a2ui(candidate).renderer_part
        for candidate in _iter_a2ui_payload_messages(response.a2ui_payload)
    )


def _redact_rejected_a2ui_payload(
    response: SpecialistResponse,
    a2ui_parts: tuple[A2APart, ...],
) -> SpecialistResponse:
    if response.a2ui_payload is None:
        return response
    if not any(isinstance(part, TextPart) for part in a2ui_parts):
        return response

    return response.model_copy(update={"a2ui_payload": None})


def _iter_a2ui_payload_messages(payload: Any) -> list[Any]:
    if isinstance(payload, Mapping):
        return [payload]
    return list(payload)


__all__ = [
    "OrchestratorRequestResult",
    "OrchestratorService",
    "OrchestratorUserActionResult",
]
