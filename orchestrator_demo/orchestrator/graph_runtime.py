"""ADK workflow runtime boundary for approved execution plans."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import importlib
from types import ModuleType
from typing import Any


ADK_WORKFLOW_MODULE = "google.adk.workflow"
START_NODE_NAME = "__START__"


class AdkGraphApiError(RuntimeError):
    """Raised when the installed ADK graph API cannot build a workflow."""


@dataclass(frozen=True)
class AdkRuntimeEdge:
    """Runtime edge definition using ADK node names and optional route tags."""

    from_node_name: str
    to_node_name: str
    route: bool | int | str | None = None


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


def _step_node_callable(node_name: str):
    async def _run_step(node_input: Any) -> dict[str, Any]:
        return {"node": node_name, "input": node_input}

    _run_step.__name__ = f"run_{node_name}"
    return _run_step


__all__ = [
    "ADK_WORKFLOW_MODULE",
    "START_NODE_NAME",
    "AdkGraphApiError",
    "AdkGraphRuntime",
    "AdkRuntimeEdge",
    "AdkWorkflowRuntimeFactory",
]
