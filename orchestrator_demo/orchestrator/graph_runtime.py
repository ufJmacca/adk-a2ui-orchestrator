"""ADK workflow runtime boundary for approved execution plans."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import importlib
import re
from types import ModuleType
from typing import Any, Protocol

from orchestrator_demo.agents import build_default_specialists
from orchestrator_demo.contracts import SpecialistRequest, SpecialistResponse
from orchestrator_demo.orchestrator.specialist_invocation import invoke_specialist


ADK_WORKFLOW_MODULE = "google.adk.workflow"
START_NODE_NAME = "__START__"
DEFAULT_ROUTE = "__DEFAULT__"
MISSING_INTERNAL_DATA_ROUTE = "missing_internal_data"
MISSING_DATA_ROUTE_KEYS = frozenset(
    {
        "missing_internal_data",
        "requires_data_quality",
        "data_quality_needed",
    }
)
EXPLICIT_ROUTE_KEYS = frozenset(
    {
        "route",
        "selected_route",
        "selectedRoute",
        "condition",
        "selected_condition",
        "selectedCondition",
        "route_condition",
        "routeCondition",
    }
)
MISSING_DATA_PHRASES = (
    "missing internal data",
    "stale internal data",
    "missing data",
    "data gap",
    "data quality gap",
    "data quality issue",
)
_NEGATIVE_MISSING_DATA_PHRASES = (
    "no missing internal data",
    "no missing data",
    "no data gaps",
    "no data quality gaps",
    "no data quality issues",
    "not missing internal data",
    "not missing data",
    "missing_internal_data=false",
    "missing_internal_data: false",
)


class AdkGraphApiError(RuntimeError):
    """Raised when the installed ADK graph API cannot build a workflow."""


class SurfaceOwnerRecorder(Protocol):
    def __call__(self, *, surface_id: str | None, agent_id: str) -> None:
        """Record deterministic ownership for a specialist-emitted surface."""


@dataclass(frozen=True)
class AdkRuntimeEdge:
    """Runtime edge definition using ADK node names and optional route tags."""

    from_node_name: str
    to_node_name: str
    route: bool | int | str | None = None


@dataclass(frozen=True)
class AdkRuntimeStep:
    """Approved plan step metadata bound to one ADK FunctionNode."""

    node_name: str
    plan_id: str
    plan_version: int
    objective: str
    step_id: str
    agent_id: str
    instruction: str
    expected_output: str
    data_source_categories: tuple[str, ...]
    depends_on: tuple[str, ...]
    parallel_group: str | None = None


@dataclass(frozen=True)
class AdkGraphRuntime:
    """ADK-backed graph object plus inspectable metadata for tests."""

    workflow: Any
    graph: Any
    node_names: tuple[str, ...]
    edge_routes: tuple[tuple[str, str, bool | int | str | None], ...]

    @property
    def is_adk_backed(self) -> bool:
        return True


class AdkWorkflowRuntimeFactory:
    """Create ADK Workflow objects without a local fallback implementation."""

    def __init__(
        self,
        *,
        workflow_module: str = ADK_WORKFLOW_MODULE,
        specialists: Mapping[str, Any] | None = None,
        record_specialist_surface_owner: SurfaceOwnerRecorder | None = None,
    ) -> None:
        self._workflow_module = workflow_module
        self._specialists = dict(specialists) if specialists is not None else None
        self._record_specialist_surface_owner = record_specialist_surface_owner

    def build(
        self,
        *,
        graph_id: str,
        step_node_names: Sequence[str],
        step_definitions: Sequence[AdkRuntimeStep],
        join_node_names: Sequence[str],
        edges: Sequence[AdkRuntimeEdge],
    ) -> AdkGraphRuntime:
        """Build and validate an ADK Workflow from graph node/edge metadata."""

        step_definitions_by_node_name = _step_definitions_by_node_name(
            step_node_names,
            step_definitions,
        )
        specialists_by_agent_id = self._specialists_by_agent_id(step_definitions)
        routes_by_node_name = _routes_by_node_name(edges)

        api = self._load_workflow_api()
        try:
            nodes = {
                node_name: api.FunctionNode(
                    func=_step_node_callable(
                        step_definitions_by_node_name[node_name],
                        specialists_by_agent_id[
                            step_definitions_by_node_name[node_name].agent_id
                        ],
                        routes_by_node_name.get(node_name, ()),
                        self._record_specialist_surface_owner,
                    ),
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

    def _specialists_by_agent_id(
        self,
        step_definitions: Sequence[AdkRuntimeStep],
    ) -> dict[str, Any]:
        specialists = (
            dict(self._specialists)
            if self._specialists is not None
            else build_default_specialists()
        )
        required_agent_ids = {step.agent_id for step in step_definitions}
        missing_agent_ids = sorted(required_agent_ids - set(specialists))
        if missing_agent_ids:
            raise AdkGraphApiError(
                "ADK workflow graph execution cannot bind specialist handlers: "
                f"missing {', '.join(missing_agent_ids)}"
            )
        return specialists

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


def _step_definitions_by_node_name(
    step_node_names: Sequence[str],
    step_definitions: Sequence[AdkRuntimeStep],
) -> dict[str, AdkRuntimeStep]:
    definitions_by_node_name = {step.node_name: step for step in step_definitions}
    if len(definitions_by_node_name) != len(step_definitions):
        raise AdkGraphApiError(
            "ADK workflow graph execution cannot bind duplicate step node definitions"
        )

    missing = [
        node_name
        for node_name in step_node_names
        if node_name not in definitions_by_node_name
    ]
    extra = [
        node_name
        for node_name in definitions_by_node_name
        if node_name not in step_node_names
    ]
    if missing or extra:
        detail = []
        if missing:
            detail.append(f"missing definitions for {', '.join(missing)}")
        if extra:
            detail.append(f"unexpected definitions for {', '.join(extra)}")
        raise AdkGraphApiError(
            "ADK workflow graph execution cannot bind approved step metadata: "
            + "; ".join(detail)
        )

    return definitions_by_node_name


def _routes_by_node_name(
    edges: Sequence[AdkRuntimeEdge],
) -> dict[str, tuple[bool | int | str, ...]]:
    routes: dict[str, list[bool | int | str]] = {}
    for edge in edges:
        if edge.route is None:
            continue
        routes.setdefault(edge.from_node_name, []).append(edge.route)
    return {
        node_name: tuple(dict.fromkeys(node_routes))
        for node_name, node_routes in routes.items()
    }


def _step_node_callable(
    step: AdkRuntimeStep,
    specialist: Any,
    route_values: Sequence[bool | int | str],
    record_specialist_surface_owner: SurfaceOwnerRecorder | None,
):
    async def _run_step(ctx: Any, node_input: Any) -> dict[str, Any]:
        upstream = _flatten_route_gate_upstream(_jsonable(node_input))
        request = SpecialistRequest(
            request_id=_request_id_for(step),
            user_input=step.instruction,
            agent_id=step.agent_id,
            plan_id=step.plan_id,
            step_id=step.step_id,
            context=_step_context(step, upstream=upstream),
        )
        response = await invoke_specialist(specialist, request)
        if record_specialist_surface_owner is not None:
            record_specialist_surface_owner(
                surface_id=response.surface_id,
                agent_id=step.agent_id,
            )

        route = _route_for_response(response, route_values)
        if route is not None:
            ctx.route = route

        return {
            "node": step.node_name,
            "planId": step.plan_id,
            "planVersion": step.plan_version,
            "stepId": step.step_id,
            "agentId": step.agent_id,
            "request": request.model_dump(mode="json"),
            "response": response.model_dump(mode="json"),
            "upstream": upstream,
        }

    _run_step.__name__ = f"run_{step.node_name}"
    return _run_step


def _flatten_route_gate_upstream(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value

    flattened = dict(value)
    for key, child_value in value.items():
        if not _is_route_gate_node_name(str(key)) or not isinstance(
            child_value, Mapping
        ):
            continue

        # Route gates are runtime-only join nodes; downstream specialists should
        # see the approved predecessor output that satisfied the gate.
        flattened.pop(key, None)
        for child_key, nested_child_value in child_value.items():
            flattened.setdefault(str(child_key), nested_child_value)
    return flattened


def _is_route_gate_node_name(value: str) -> bool:
    return value.startswith("join_") and value.endswith("_gate")


def _step_context(step: AdkRuntimeStep, *, upstream: Any = None) -> dict[str, Any]:
    context: dict[str, Any] = {
        "objective": step.objective,
        "planVersion": step.plan_version,
        "expectedOutput": step.expected_output,
        "dataSourceCategories": list(step.data_source_categories),
        "dependsOn": list(step.depends_on),
    }
    if step.depends_on:
        context["upstream"] = upstream
    if step.parallel_group is not None:
        context["parallelGroup"] = step.parallel_group
    return context


def _route_for_response(
    response: SpecialistResponse,
    route_values: Sequence[bool | int | str],
) -> bool | int | str | None:
    if not route_values:
        return None
    explicit_route = _explicit_route_key(response.structured_output)
    if explicit_route in route_values:
        return explicit_route
    if (
        MISSING_INTERNAL_DATA_ROUTE in route_values
        and _execution_indicates_missing_data(response)
    ):
        return MISSING_INTERNAL_DATA_ROUTE
    for route_value in route_values:
        if route_value in {DEFAULT_ROUTE, MISSING_INTERNAL_DATA_ROUTE}:
            continue
        if _execution_indicates_route(response, route_value):
            return route_value
    if DEFAULT_ROUTE in route_values:
        return DEFAULT_ROUTE
    return None


def _execution_indicates_route(
    response: SpecialistResponse,
    route_value: bool | int | str,
) -> bool:
    explicit_route = _explicit_route_key(response.structured_output)
    if explicit_route is not None:
        return explicit_route == route_value

    if isinstance(route_value, str):
        explicit_flag = _explicit_bool_key(
            response.structured_output,
            frozenset({route_value}),
        )
        if explicit_flag is not None:
            return explicit_flag

    return False


def _explicit_route_key(value: Any) -> bool | int | str | None:
    if isinstance(value, Mapping):
        for key, child_value in value.items():
            if (
                isinstance(key, str)
                and key in EXPLICIT_ROUTE_KEYS
                and isinstance(child_value, bool | int | str)
            ):
                return child_value
            child_result = _explicit_route_key(child_value)
            if child_result is not None:
                return child_result
    elif isinstance(value, list | tuple):
        for item in value:
            child_result = _explicit_route_key(item)
            if child_result is not None:
                return child_result
    return None


def _execution_indicates_missing_data(
    response: SpecialistResponse,
) -> bool:
    explicit_flag = _explicit_bool_key(
        response.structured_output,
        MISSING_DATA_ROUTE_KEYS,
    )
    if explicit_flag is not None:
        return explicit_flag

    searchable_text = " ".join(
        [
            response.content,
            *_text_values(response.structured_output),
        ]
    )
    return _text_indicates_missing_data(searchable_text)


def _explicit_bool_key(value: Any, keys: frozenset[str]) -> bool | None:
    if isinstance(value, Mapping):
        for key, child_value in value.items():
            if isinstance(key, str) and key in keys:
                explicit_bool = _coerce_explicit_bool(child_value)
                if explicit_bool is not None:
                    return explicit_bool
                continue
            child_result = _explicit_bool_key(child_value, keys)
            if child_result is not None:
                return child_result
    elif isinstance(value, list | tuple):
        for item in value:
            child_result = _explicit_bool_key(item, keys)
            if child_result is not None:
                return child_result
    return None


def _coerce_explicit_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {
            "false",
            "no",
            "none",
            "0",
            "not required",
            "not_required",
            "not needed",
            "not_needed",
            "no missing data",
            "no missing internal data",
        }:
            return False
        if normalized in {"true", "yes", "1", "required"}:
            return True
        return _missing_data_signal(normalized)
    if isinstance(value, Mapping):
        for child_value in value.values():
            child_result = _coerce_explicit_bool(child_value)
            if child_result is not None:
                return child_result
    elif isinstance(value, list | tuple):
        for item in value:
            child_result = _coerce_explicit_bool(item)
            if child_result is not None:
                return child_result
    return None


def _text_indicates_missing_data(searchable_text: str) -> bool:
    return _missing_data_signal(searchable_text) is True


def _missing_data_signal(searchable_text: str) -> bool | None:
    normalized = searchable_text.lower()
    if any(phrase in normalized for phrase in _NEGATIVE_MISSING_DATA_PHRASES):
        return False
    if any(phrase in normalized for phrase in MISSING_DATA_PHRASES):
        return True
    return None


def _text_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        texts: list[str] = []
        for child_value in value.values():
            texts.extend(_text_values(child_value))
        return texts
    if isinstance(value, list | tuple):
        texts = []
        for item in value:
            texts.extend(_text_values(item))
        return texts
    return []


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child_value) for key, child_value in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump(mode="json")
        except TypeError:
            return model_dump()

    return repr(value)


def _request_id_for(step: AdkRuntimeStep) -> str:
    return (
        f"request_{_identifier_suffix(step.plan_id)}_"
        f"{_identifier_suffix(step.step_id)}"
    )


def _identifier_suffix(value: str) -> str:
    identifier = re.sub(r"[^A-Za-z0-9_]+", "_", value.strip())
    identifier = identifier.strip("_")
    if not identifier:
        return "unknown"
    if identifier[0].isdigit():
        return f"id_{identifier}"
    return identifier


__all__ = [
    "ADK_WORKFLOW_MODULE",
    "START_NODE_NAME",
    "AdkGraphApiError",
    "AdkGraphRuntime",
    "AdkRuntimeEdge",
    "AdkRuntimeStep",
    "AdkWorkflowRuntimeFactory",
]
