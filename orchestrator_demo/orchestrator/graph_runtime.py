"""ADK-backed graph construction and execution for approved plans."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import threading
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Protocol

from orchestrator_demo.app.logging import log_audit_event
from orchestrator_demo.contracts import (
    ExecutionPlan,
    GraphEdge,
    GraphPattern,
    GraphSpec,
    GraphStep,
    PlanStep,
    SpecialistRequest,
    SpecialistResponse,
    StatusEvent,
    StatusName,
)


ADK_WORKFLOW_MODULE = "google.adk.workflow"
START_NODE_NAME = "__START__"

SpecialistStepHandler = Callable[
    [SpecialistRequest],
    SpecialistResponse | Awaitable[SpecialistResponse],
]


class GraphRuntimeError(RuntimeError):
    """Raised when ADK graph construction or execution fails."""

    def __init__(
        self,
        message: str,
        *,
        graph: GraphSpec | None = None,
        status_events: Sequence[StatusEvent] = (),
        specialist_requests: Sequence[SpecialistRequest] = (),
        specialist_responses: Sequence[SpecialistResponse] = (),
        adk_event_outputs: Sequence[Any] = (),
    ) -> None:
        super().__init__(message)
        self.graph = graph
        self.status_events = tuple(status_events)
        self.specialist_requests = tuple(specialist_requests)
        self.specialist_responses = tuple(specialist_responses)
        self.adk_event_outputs = tuple(adk_event_outputs)


class AdkGraphApiError(RuntimeError):
    """Raised when the installed ADK graph API cannot build a workflow."""


@dataclass(frozen=True)
class GraphExecutionResult:
    """Observable output from executing an approved plan graph."""

    graph: GraphSpec
    workflow: Any
    status_events: tuple[StatusEvent, ...]
    specialist_requests: tuple[SpecialistRequest, ...]
    specialist_responses: tuple[SpecialistResponse, ...]
    adk_event_outputs: tuple[Any, ...]


@dataclass(frozen=True)
class AdkRuntimeEdge:
    """Runtime edge definition using ADK node names and optional route tags."""

    from_node_name: str
    to_node_name: str
    route: bool | int | str | None = None


class GraphRuntime(Protocol):
    """Runtime interface consumed by approval state."""

    def execute(self, plan: ExecutionPlan) -> GraphExecutionResult:
        """Create and execute the approved plan graph."""
        ...


class AdkGraphRuntime:
    """ADK-backed execution runtime or inspectable built workflow metadata."""

    def __init__(
        self,
        *,
        specialist_handlers: Mapping[str, SpecialistStepHandler] | None = None,
        workflow: Any | None = None,
        graph: Any | None = None,
        node_names: Sequence[str] = (),
        edge_routes: Sequence[tuple[str, str, bool | int | str | None]] = (),
    ) -> None:
        self._specialist_handlers = dict(specialist_handlers or {})
        self.workflow = workflow
        self.graph = graph
        self.node_names = tuple(node_names)
        self.edge_routes = tuple(edge_routes)

    @property
    def is_adk_backed(self) -> bool:
        return True

    def execute(self, plan: ExecutionPlan) -> GraphExecutionResult:
        """Create an ADK workflow for the approved plan and execute every step."""

        graph = build_graph_spec(plan)
        log_audit_event(
            "graph_execution_started",
            {
                "graph_id": graph.graph_id,
                "plan_id": graph.plan_id,
                "pattern": graph.pattern,
                "step_count": len(graph.steps),
            },
        )
        log_audit_event(
            "graph_created",
            {
                "graph_id": graph.graph_id,
                "plan_id": graph.plan_id,
                "pattern": graph.pattern,
                "step_count": len(graph.steps),
                "edge_count": len(graph.edges),
            },
        )
        events: list[StatusEvent] = [
            _status_event(
                graph,
                "plan_approved",
                "plan_approved",
                f"Plan {plan.plan_id} approved for graph execution.",
                details={"planVersion": plan.plan_version},
            ),
            _status_event(
                graph,
                "graph_created",
                "graph_created",
                f"ADK graph created for approved plan {plan.plan_id}.",
            ),
        ]
        requests: list[SpecialistRequest] = []
        responses: list[SpecialistResponse] = []
        step_outputs: dict[str, dict[str, Any]] = {}
        _raise_for_missing_specialist_handlers(
            plan=plan,
            graph=graph,
            specialist_handlers=self._specialist_handlers,
            events=events,
            requests=requests,
            responses=responses,
        )

        try:
            workflow = self._build_workflow(
                plan=plan,
                graph=graph,
                events=events,
                requests=requests,
                responses=responses,
                step_outputs=step_outputs,
            )
            outputs = _run_coroutine_blocking(_collect_adk_outputs(workflow, plan))
        except Exception as exc:
            if isinstance(exc, GraphRuntimeError):
                log_audit_event(
                    "graph_execution_failed",
                    {
                        "graph_id": graph.graph_id,
                        "plan_id": graph.plan_id,
                        "error_type": type(exc).__name__,
                        "status_event_count": len(events),
                        "request_count": len(requests),
                        "response_count": len(responses),
                    },
                )
                raise GraphRuntimeError(
                    _execution_failure_message(exc, events),
                    graph=exc.graph or graph,
                    status_events=exc.status_events or events,
                    specialist_requests=exc.specialist_requests or requests,
                    specialist_responses=exc.specialist_responses or responses,
                    adk_event_outputs=exc.adk_event_outputs,
                ) from exc
            log_audit_event(
                "graph_execution_failed",
                {
                    "graph_id": graph.graph_id,
                    "plan_id": graph.plan_id,
                    "error_type": type(exc).__name__,
                    "status_event_count": len(events),
                    "request_count": len(requests),
                    "response_count": len(responses),
                },
            )
            raise GraphRuntimeError(
                f"ADK graph execution failed: {type(exc).__name__}",
                graph=graph,
                status_events=events,
                specialist_requests=requests,
                specialist_responses=responses,
            ) from exc

        events.append(
            _status_event(
                graph,
                "final_response_ready",
                "final_response_ready",
                f"Approved plan {plan.plan_id} execution completed.",
                details={"responseCount": len(responses)},
            )
        )
        log_audit_event(
            "graph_execution_completed",
            {
                "graph_id": graph.graph_id,
                "plan_id": graph.plan_id,
                "status_event_count": len(events),
                "request_count": len(requests),
                "response_count": len(responses),
            },
        )
        return GraphExecutionResult(
            graph=graph,
            workflow=workflow,
            status_events=tuple(events),
            specialist_requests=tuple(requests),
            specialist_responses=tuple(responses),
            adk_event_outputs=tuple(outputs),
        )

    def _build_workflow(
        self,
        *,
        plan: ExecutionPlan,
        graph: GraphSpec,
        events: list[StatusEvent],
        requests: list[SpecialistRequest],
        responses: list[SpecialistResponse],
        step_outputs: dict[str, dict[str, Any]],
    ) -> Any:
        workflow_api = _adk_workflow_api()
        graph_step_ids = {
            step.plan_step_id: step.graph_step_id for step in graph.steps
        }
        nodes_by_step_id = {
            step.step_id: workflow_api.FunctionNode(
                func=_step_function(
                    plan=plan,
                    step=step,
                    graph=graph,
                    graph_step_id=graph_step_ids[step.step_id],
                    events=events,
                    requests=requests,
                    responses=responses,
                    step_outputs=step_outputs,
                    specialist_handlers=self._specialist_handlers,
                ),
                name=_node_name("node", step.step_id),
            )
            for step in plan.steps
        }

        edges: list[Any] = []
        for step in plan.steps:
            node = nodes_by_step_id[step.step_id]
            if not step.depends_on:
                edges.append(
                    workflow_api.Edge(from_node=workflow_api.START, to_node=node)
                )
                continue

            dependency_nodes = [
                nodes_by_step_id[dependency_step_id]
                for dependency_step_id in step.depends_on
            ]
            if len(dependency_nodes) == 1:
                edges.append(
                    workflow_api.Edge(from_node=dependency_nodes[0], to_node=node)
                )
                continue

            join_node = workflow_api.JoinNode(name=_node_name("join", step.step_id))
            for dependency_node in dependency_nodes:
                edges.append(
                    workflow_api.Edge(from_node=dependency_node, to_node=join_node)
                )
            edges.append(workflow_api.Edge(from_node=join_node, to_node=node))

        if not edges:
            raise GraphRuntimeError("approved plan graph has no executable edges")

        return workflow_api.Workflow(
            name=_node_name("workflow", plan.plan_id),
            edges=edges,
        )


class AdkWorkflowRuntimeFactory:
    """Create ADK Workflow objects without a local fallback implementation."""

    def __init__(self, *, workflow_module: str = ADK_WORKFLOW_MODULE) -> None:
        self._workflow_module = workflow_module

    def build(
        self,
        *,
        graph_id: str,
        step_node_names: Sequence[str],
        join_node_names: Sequence[str],
        edges: Sequence[AdkRuntimeEdge],
    ) -> AdkGraphRuntime:
        """Build and validate an ADK Workflow from graph node/edge metadata."""

        api = self._load_workflow_api()
        try:
            nodes = {
                node_name: api.FunctionNode(
                    func=_step_node_callable(node_name),
                    name=node_name,
                )
                for node_name in step_node_names
            }
            nodes.update(
                {
                    node_name: api.JoinNode(name=node_name)
                    for node_name in join_node_names
                }
            )

            adk_edges = [
                api.Edge(
                    from_node=(
                        api.START
                        if edge.from_node_name == START_NODE_NAME
                        else nodes[edge.from_node_name]
                    ),
                    to_node=nodes[edge.to_node_name],
                    route=edge.route,
                )
                for edge in edges
            ]
            workflow = api.Workflow(name=graph_id, edges=adk_edges)
        except Exception as exc:
            raise AdkGraphApiError(
                "ADK workflow graph API unavailable or incompatible: "
                f"failed to instantiate workflow {graph_id!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        if getattr(workflow, "graph", None) is None:
            raise AdkGraphApiError(
                "ADK workflow graph API unavailable or incompatible: "
                f"workflow {graph_id!r} did not expose a compiled graph"
            )

        try:
            graph = workflow.graph
            node_names = tuple(node.name for node in graph.nodes)
            edge_routes = tuple(
                (edge.from_node.name, edge.to_node.name, edge.route)
                for edge in graph.edges
            )
        except Exception as exc:
            raise AdkGraphApiError(
                "ADK workflow graph API unavailable or incompatible: "
                f"workflow {graph_id!r} exposed an incompatible compiled graph: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        return AdkGraphRuntime(
            workflow=workflow,
            graph=graph,
            node_names=node_names,
            edge_routes=edge_routes,
        )

    def _load_workflow_api(self) -> ModuleType:
        try:
            api = importlib.import_module(self._workflow_module)
        except Exception as exc:
            raise AdkGraphApiError(
                "ADK workflow graph API unavailable or incompatible: "
                f"cannot import {self._workflow_module!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        required_attributes = ["Workflow", "FunctionNode", "JoinNode", "Edge", "START"]
        missing = [
            attribute
            for attribute in required_attributes
            if not hasattr(api, attribute)
        ]
        if missing:
            raise AdkGraphApiError(
                "ADK workflow graph API unavailable or incompatible: "
                f"{self._workflow_module!r} is missing {', '.join(missing)}"
            )

        return api


def default_specialist_handlers(
    agent_ids: Iterable[str] | None = None,
) -> dict[str, SpecialistStepHandler]:
    """Create default local specialist handlers, optionally scoped to registered ids."""

    from orchestrator_demo.agents import build_default_specialists

    allowed_agent_ids = set(agent_ids) if agent_ids is not None else None
    specialists = build_default_specialists()
    return {
        agent_id: specialist.handle
        for agent_id, specialist in specialists.items()
        if allowed_agent_ids is None or agent_id in allowed_agent_ids
    }


def _missing_specialist_handler_steps(
    plan: ExecutionPlan,
    specialist_handlers: Mapping[str, SpecialistStepHandler],
) -> list[PlanStep]:
    return [
        step for step in plan.steps if step.agent_id not in specialist_handlers
    ]


def _raise_for_missing_specialist_handlers(
    *,
    plan: ExecutionPlan,
    graph: GraphSpec,
    specialist_handlers: Mapping[str, SpecialistStepHandler],
    events: list[StatusEvent],
    requests: list[SpecialistRequest],
    responses: list[SpecialistResponse],
) -> None:
    missing_steps = _missing_specialist_handler_steps(plan, specialist_handlers)
    if not missing_steps:
        return

    for step in missing_steps:
        graph_step = _graph_step_id(step.step_id)
        events.append(
            _status_event(
                graph,
                f"{graph_step}_failed",
                "step_failed",
                (
                    f"Approved plan step {step.step_id} failed before execution: "
                    "no specialist handler registered for agent "
                    f"{step.agent_id}."
                ),
                step_id=graph_step,
                details={
                    "agentId": step.agent_id,
                    "planStepId": step.step_id,
                    "developerMessage": (
                        f"Register agent {step.agent_id} before executing "
                        f"approved plan {plan.plan_id}."
                    ),
                },
            )
        )

    raise GraphRuntimeError(
        _missing_specialist_handlers_message(missing_steps),
        graph=graph,
        status_events=events,
        specialist_requests=requests,
        specialist_responses=responses,
    )


def _missing_specialist_handlers_message(missing_steps: Sequence[PlanStep]) -> str:
    if len(missing_steps) == 1:
        step = missing_steps[0]
        return (
            "no specialist handler registered for approved plan step "
            f"{step.step_id} agent {step.agent_id}"
        )

    missing = ", ".join(
        f"{step.step_id} agent {step.agent_id}" for step in missing_steps
    )
    return (
        "no specialist handlers registered for approved plan steps: "
        f"{missing}"
    )


@dataclass(frozen=True)
class _WorkflowApi:
    Workflow: Any
    FunctionNode: Any
    JoinNode: Any
    Edge: Any
    START: Any


def _adk_workflow_api() -> _WorkflowApi:
    try:
        from google.adk.workflow import Edge, FunctionNode, JoinNode, START, Workflow
    except Exception as exc:
        raise GraphRuntimeError(
            "ADK graph workflow APIs are unavailable or incompatible"
        ) from exc

    return _WorkflowApi(
        Workflow=Workflow,
        FunctionNode=FunctionNode,
        JoinNode=JoinNode,
        Edge=Edge,
        START=START,
    )


def build_graph_spec(plan: ExecutionPlan) -> GraphSpec:
    """Convert an approved execution plan into the local graph contract."""

    graph_steps = [
        GraphStep(
            graph_step_id=_graph_step_id(step.step_id),
            plan_step_id=step.step_id,
            agent_id=step.agent_id,
            depends_on=[_graph_step_id(dependency) for dependency in step.depends_on],
            parallel_group=step.parallel_group,
        )
        for step in plan.steps
    ]
    graph_edges = [
        GraphEdge(
            from_step_id=_graph_step_id(dependency),
            to_step_id=_graph_step_id(step.step_id),
        )
        for step in plan.steps
        for dependency in step.depends_on
    ]

    return GraphSpec(
        graph_id=f"graph_{plan.plan_id.removeprefix('plan_')}",
        plan_id=plan.plan_id,
        pattern=_graph_pattern(plan),
        steps=graph_steps,
        edges=graph_edges,
    )


def _step_function(
    *,
    plan: ExecutionPlan,
    step: PlanStep,
    graph: GraphSpec,
    graph_step_id: str,
    events: list[StatusEvent],
    requests: list[SpecialistRequest],
    responses: list[SpecialistResponse],
    step_outputs: dict[str, dict[str, Any]],
    specialist_handlers: Mapping[str, SpecialistStepHandler],
) -> Callable[[], Awaitable[dict[str, Any]]]:
    async def run_step() -> dict[str, Any]:
        if step.parallel_group is not None:
            events.append(
                _status_event(
                    graph,
                    f"{graph_step_id}_parallel_started",
                    "parallel_branch_started",
                    f"Parallel branch {step.parallel_group} started.",
                    step_id=graph_step_id,
                    details={"parallelGroup": step.parallel_group},
                )
            )
        if step.agent_id == "synthesis":
            events.append(
                _status_event(
                    graph,
                    f"{graph_step_id}_synthesis_started",
                    "synthesis_started",
                    "Synthesis started.",
                    step_id=graph_step_id,
                )
            )

        events.append(
            _status_event(
                graph,
                f"{graph_step_id}_started",
                "step_started",
                f"Step {step.step_id} started.",
                step_id=graph_step_id,
            )
        )
        request = SpecialistRequest(
            request_id=_request_id(plan.plan_id, step.step_id),
            user_input=step.instruction,
            agent_id=step.agent_id,
            plan_id=plan.plan_id,
            step_id=step.step_id,
            context=_step_context(plan, step, step_outputs),
        )
        requests.append(request)
        handler = specialist_handlers.get(step.agent_id)
        if handler is None:
            raise GraphRuntimeError(
                "no specialist handler registered for approved plan step "
                f"{step.step_id} agent {step.agent_id}"
            )
        try:
            response = handler(request)
            if inspect.isawaitable(response):
                response = await response
            output = response.model_dump(mode="json")
        except Exception as exc:
            error_type = type(exc).__name__
        else:
            responses.append(response)
            step_outputs[step.step_id] = output
            events.append(
                _status_event(
                    graph,
                    f"{graph_step_id}_completed",
                    "step_completed",
                    f"Step {step.step_id} completed.",
                    step_id=graph_step_id,
                    details={"agentId": step.agent_id},
                )
            )
            if step.parallel_group is not None:
                events.append(
                    _status_event(
                        graph,
                        f"{graph_step_id}_parallel_completed",
                        "parallel_branch_completed",
                        f"Parallel branch {step.parallel_group} completed.",
                        step_id=graph_step_id,
                        details={"parallelGroup": step.parallel_group},
                    )
                )
            return output

        events.append(
            _step_failed_event(
                plan=plan,
                step=step,
                graph=graph,
                graph_step_id=graph_step_id,
                error_type=error_type,
            )
        )
        raise GraphRuntimeError(
            _handler_failure_message(plan=plan, step=step, error_type=error_type),
            graph=graph,
            status_events=events,
            specialist_requests=requests,
            specialist_responses=responses,
        )

    return run_step


async def _collect_adk_outputs(workflow: Any, plan: ExecutionPlan) -> list[Any]:
    try:
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from google.genai import types
    except Exception as exc:
        raise GraphRuntimeError(
            "ADK runner APIs are unavailable or incompatible"
        ) from exc

    session_service = InMemorySessionService()
    app_name = "orchestrator_demo"
    user_id = "local_operator"
    session_id = f"session_{plan.plan_id.removeprefix('plan_')}"
    await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
    )
    runner = Runner(app_name=app_name, node=workflow, session_service=session_service)
    new_message = types.Content(
        role="user",
        parts=[types.Part(text=plan.objective)],
    )
    outputs: list[Any] = []
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=new_message,
    ):
        if event.error_code:
            raise GraphRuntimeError(
                f"ADK graph execution failed at event {event.id}: "
                f"{event.error_code}"
            )
        if event.output is not None:
            outputs.append(event.output)

    return outputs


def _run_coroutine_blocking(coroutine: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    result: list[Any] = []
    errors: list[BaseException] = []

    def run_in_thread() -> None:
        try:
            result.append(asyncio.run(coroutine))
        except BaseException as exc:  # pragma: no cover - re-raised in caller.
            errors.append(exc)

    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()
    thread.join()
    if errors:
        raise errors[0]
    return result[0]


def _step_context(
    plan: ExecutionPlan,
    step: PlanStep,
    step_outputs: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    context: dict[str, Any] = {
        "objective": plan.objective,
        "planVersion": plan.plan_version,
        "expectedOutput": step.expected_output,
        "dataSourceCategories": list(step.data_source_categories),
        "dependsOn": list(step.depends_on),
    }
    if step.parallel_group is not None:
        context["parallelGroup"] = step.parallel_group
    if step.depends_on:
        dependency_outputs = {
            step_id: step_outputs[step_id]
            for step_id in step.depends_on
            if step_id in step_outputs
        }
        context["dependencyOutputs"] = dependency_outputs
        context["stepResults"] = dependency_outputs

    return context


def _execution_failure_message(
    exc: GraphRuntimeError,
    events: Sequence[StatusEvent],
) -> str:
    failed_event = next(
        (event for event in reversed(events) if event.status == "step_failed"),
        None,
    )
    if failed_event is not None:
        return failed_event.message
    return str(exc)


def _step_failed_event(
    *,
    plan: ExecutionPlan,
    step: PlanStep,
    graph: GraphSpec,
    graph_step_id: str,
    error_type: str,
) -> StatusEvent:
    return _status_event(
        graph,
        f"{graph_step_id}_failed",
        "step_failed",
        (
            f"Approved plan step {step.step_id} failed during execution: "
            f"{error_type}."
        ),
        step_id=graph_step_id,
        details={
            "agentId": step.agent_id,
            "planStepId": step.step_id,
            "errorType": error_type,
            "developerMessage": (
                f"Specialist handler for agent {step.agent_id} raised {error_type} "
                f"while executing approved plan "
                f"{plan.plan_id} step {step.step_id}."
            ),
        },
    )


def _handler_failure_message(
    *,
    plan: ExecutionPlan,
    step: PlanStep,
    error_type: str,
) -> str:
    return (
        f"specialist handler for approved plan {plan.plan_id} step "
        f"{step.step_id} agent {step.agent_id} failed: "
        f"{error_type}"
    )


def _graph_pattern(plan: ExecutionPlan) -> GraphPattern:
    if len(plan.steps) == 1:
        return "direct"
    if any(len(step.depends_on) > 1 for step in plan.steps):
        return "fan_out_fan_in"
    if any(step.parallel_group is not None for step in plan.steps):
        return "mixed"
    if any(step.depends_on for step in plan.steps):
        return "sequential"
    return "sequential"


def _graph_step_id(step_id: str) -> str:
    return f"graph_{step_id}"


def _request_id(plan_id: str, step_id: str) -> str:
    return f"request_{plan_id}_{step_id}"


def _node_name(prefix: str, value: str) -> str:
    return f"{prefix}_{value}".replace("-", "_")


def _step_node_callable(node_name: str):
    async def _run_step(node_input: Any) -> dict[str, Any]:
        return {"node": node_name, "input": node_input}

    _run_step.__name__ = f"run_{node_name}"
    return _run_step


def _status_event(
    graph: GraphSpec,
    suffix: str,
    status: StatusName,
    message: str,
    *,
    step_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> StatusEvent:
    return StatusEvent(
        event_id=f"event_{graph.graph_id}_{suffix}",
        graph_id=graph.graph_id,
        plan_id=graph.plan_id,
        step_id=step_id,
        status=status,
        message=message,
        details=details or {},
    )


__all__ = [
    "ADK_WORKFLOW_MODULE",
    "START_NODE_NAME",
    "AdkGraphApiError",
    "AdkGraphRuntime",
    "AdkRuntimeEdge",
    "AdkWorkflowRuntimeFactory",
    "GraphExecutionResult",
    "GraphRuntime",
    "GraphRuntimeError",
    "build_graph_spec",
    "default_specialist_handlers",
]
