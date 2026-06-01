"""Build ADK-backed graph workflows from approved immutable plans."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
import re
from typing import Protocol

from orchestrator_demo.contracts import (
    ExecutionPlan,
    GraphEdge,
    GraphPattern,
    GraphSpec,
    GraphStep,
    PlanStep,
)
from orchestrator_demo.orchestrator.approval_state import (
    ApprovalActionResult,
    ApprovalRecord,
)
from orchestrator_demo.orchestrator.graph_runtime import (
    AdkGraphRuntime,
    AdkRuntimeEdge,
    AdkWorkflowRuntimeFactory,
    START_NODE_NAME,
)


DEFAULT_ROUTE = "__DEFAULT__"
MISSING_INTERNAL_DATA_ROUTE = "missing_internal_data"


class GraphBuildError(RuntimeError):
    """Base error for graph construction failures."""


class GraphPlanApprovalError(GraphBuildError):
    """Raised when graph creation is attempted without approved immutable state."""


class GraphAgentRegistry(Protocol):
    def require_plan_agents_available(self, plan: ExecutionPlan) -> None:
        """Raise when an approved plan references unavailable specialists."""


@dataclass(frozen=True)
class BuiltGraph:
    """Serializable graph spec and the ADK-backed runtime object."""

    spec: GraphSpec
    runtime: AdkGraphRuntime


@dataclass(frozen=True)
class _ConditionalBranch:
    source_step_id: str
    data_quality_step_id: str
    synthesis_step_id: str


class GraphBuilder:
    """Convert approved plans into serializable specs and ADK workflows."""

    def __init__(
        self,
        *,
        registry: GraphAgentRegistry,
        runtime_factory: AdkWorkflowRuntimeFactory | None = None,
    ) -> None:
        self._registry = registry
        self._runtime_factory = runtime_factory or AdkWorkflowRuntimeFactory()

    def build(self, approval: ApprovalRecord | ApprovalActionResult) -> BuiltGraph:
        """Create a graph only from an approved immutable plan."""

        plan = _approved_immutable_plan(approval)
        self._registry.require_plan_agents_available(plan)

        graph_step_ids_by_plan_step_id = _graph_step_ids_by_plan_step_id(plan.steps)
        conditional_branch = _conditional_data_quality_branch(plan)
        spec = _graph_spec_for(
            plan,
            graph_step_ids_by_plan_step_id=graph_step_ids_by_plan_step_id,
            conditional_branch=conditional_branch,
        )
        runtime_edges, join_node_names = _runtime_edges_for(
            spec,
            root_graph_step_ids=_root_graph_step_ids(
                plan.steps,
                graph_step_ids_by_plan_step_id=graph_step_ids_by_plan_step_id,
            ),
        )
        runtime = self._runtime_factory.build(
            graph_id=spec.graph_id,
            step_node_names=[step.graph_step_id for step in spec.steps],
            join_node_names=join_node_names,
            edges=runtime_edges,
        )
        return BuiltGraph(spec=spec, runtime=runtime)


def _approved_immutable_plan(
    approval: ApprovalRecord | ApprovalActionResult,
) -> ExecutionPlan:
    if isinstance(approval, ApprovalRecord):
        if approval.status != "approved" or approval.approved_plan is None:
            raise GraphPlanApprovalError(
                "GraphBuilder requires an approved immutable plan record"
            )
        plan = approval.approved_plan
    elif isinstance(approval, ApprovalActionResult):
        if approval.status != "approved" or approval.approved_plan is None:
            raise GraphPlanApprovalError(
                "GraphBuilder requires an approved immutable plan result"
            )
        plan = approval.approved_plan.model_copy(deep=True)
    else:
        raise GraphPlanApprovalError(
            "GraphBuilder requires an approved immutable plan record or result"
        )

    if plan.immutable_after_approval is not True:
        raise GraphPlanApprovalError(
            "approved plan must set immutable_after_approval=True before graph creation"
        )

    return plan


def _graph_spec_for(
    plan: ExecutionPlan,
    *,
    graph_step_ids_by_plan_step_id: dict[str, str],
    conditional_branch: _ConditionalBranch | None,
) -> GraphSpec:
    graph_steps = [
        GraphStep(
            graph_step_id=graph_step_ids_by_plan_step_id[step.step_id],
            plan_step_id=step.step_id,
            agent_id=step.agent_id,
            depends_on=[
                graph_step_ids_by_plan_step_id[dependency_id]
                for dependency_id in step.depends_on
            ],
            parallel_group=step.parallel_group,
        )
        for step in plan.steps
    ]

    return GraphSpec(
        graph_id=_graph_id_for(plan.plan_id),
        plan_id=plan.plan_id,
        pattern=_graph_pattern_for(plan, conditional_branch=conditional_branch),
        steps=graph_steps,
        edges=_graph_edges_for(
            plan.steps,
            graph_step_ids_by_plan_step_id=graph_step_ids_by_plan_step_id,
            conditional_branch=conditional_branch,
        ),
    )


def _graph_edges_for(
    steps: Sequence[PlanStep],
    *,
    graph_step_ids_by_plan_step_id: dict[str, str],
    conditional_branch: _ConditionalBranch | None,
) -> list[GraphEdge]:
    edges: list[GraphEdge] = []
    for step in steps:
        for dependency_id in step.depends_on:
            condition: str | None = None
            if conditional_branch is not None:
                condition = _conditional_edge_condition(
                    dependency_id=dependency_id,
                    step_id=step.step_id,
                    conditional_branch=conditional_branch,
                )
            _append_graph_edge(
                edges,
                from_step_id=graph_step_ids_by_plan_step_id[dependency_id],
                to_step_id=graph_step_ids_by_plan_step_id[step.step_id],
                condition=condition,
            )

    if conditional_branch is not None:
        _append_graph_edge(
            edges,
            from_step_id=graph_step_ids_by_plan_step_id[
                conditional_branch.source_step_id
            ],
            to_step_id=graph_step_ids_by_plan_step_id[
                conditional_branch.synthesis_step_id
            ],
            condition=DEFAULT_ROUTE,
        )

    return edges


def _conditional_edge_condition(
    *,
    dependency_id: str,
    step_id: str,
    conditional_branch: _ConditionalBranch,
) -> str | None:
    if (
        dependency_id == conditional_branch.source_step_id
        and step_id == conditional_branch.data_quality_step_id
    ):
        return MISSING_INTERNAL_DATA_ROUTE
    if (
        dependency_id == conditional_branch.source_step_id
        and step_id == conditional_branch.synthesis_step_id
    ):
        return DEFAULT_ROUTE
    return None


def _append_graph_edge(
    edges: list[GraphEdge],
    *,
    from_step_id: str,
    to_step_id: str,
    condition: str | None,
) -> None:
    candidate = GraphEdge(
        from_step_id=from_step_id,
        to_step_id=to_step_id,
        condition=condition,
    )
    if candidate not in edges:
        edges.append(candidate)


def _runtime_edges_for(
    spec: GraphSpec,
    *,
    root_graph_step_ids: Sequence[str],
) -> tuple[list[AdkRuntimeEdge], list[str]]:
    graph_edges = list(spec.edges)
    incoming_edges_by_target = _incoming_edges_by_target(graph_edges)
    join_targets = {
        target_id
        for target_id, incoming_edges in incoming_edges_by_target.items()
        if len(incoming_edges) > 1
        and all(edge.condition is None for edge in incoming_edges)
    }
    join_node_names = [f"join_{target_id}" for target_id in _ordered_ids(join_targets)]
    join_node_by_target = dict(zip(_ordered_ids(join_targets), join_node_names))

    runtime_edges: list[AdkRuntimeEdge] = []
    for graph_step_id in root_graph_step_ids:
        _append_runtime_edge(
            runtime_edges,
            AdkRuntimeEdge(
                from_node_name=START_NODE_NAME,
                to_node_name=graph_step_id,
            ),
        )

    for edge in graph_edges:
        target_join = join_node_by_target.get(edge.to_step_id)
        if target_join is not None:
            _append_runtime_edge(
                runtime_edges,
                AdkRuntimeEdge(
                    from_node_name=edge.from_step_id,
                    to_node_name=target_join,
                    route=edge.condition,
                ),
            )
            continue
        _append_runtime_edge(
            runtime_edges,
            AdkRuntimeEdge(
                from_node_name=edge.from_step_id,
                to_node_name=edge.to_step_id,
                route=edge.condition,
            ),
        )

    for target_id in _ordered_ids(join_targets):
        _append_runtime_edge(
            runtime_edges,
            AdkRuntimeEdge(
                from_node_name=join_node_by_target[target_id],
                to_node_name=target_id,
            ),
        )

    return runtime_edges, join_node_names


def _append_runtime_edge(
    edges: list[AdkRuntimeEdge],
    candidate: AdkRuntimeEdge,
) -> None:
    if candidate not in edges:
        edges.append(candidate)


def _incoming_edges_by_target(
    edges: Sequence[GraphEdge],
) -> dict[str, list[GraphEdge]]:
    incoming_edges: dict[str, list[GraphEdge]] = {}
    for edge in edges:
        incoming_edges.setdefault(edge.to_step_id, []).append(edge)
    return incoming_edges


def _root_graph_step_ids(
    steps: Sequence[PlanStep],
    *,
    graph_step_ids_by_plan_step_id: dict[str, str],
) -> list[str]:
    return [
        graph_step_ids_by_plan_step_id[step.step_id]
        for step in steps
        if not step.depends_on
    ]


def _graph_pattern_for(
    plan: ExecutionPlan,
    *,
    conditional_branch: _ConditionalBranch | None,
) -> GraphPattern:
    if conditional_branch is not None:
        return "conditional"
    if len(plan.steps) == 1:
        return "direct"
    if _has_mixed_sequential_parallel(plan.steps):
        return "mixed"
    if _has_fan_out_fan_in(plan.steps):
        return "fan_out_fan_in"
    return "sequential"


def _has_mixed_sequential_parallel(steps: Sequence[PlanStep]) -> bool:
    parallel_group_counts = Counter(
        step.parallel_group for step in steps if step.parallel_group is not None
    )
    has_parallel_group = any(count > 1 for count in parallel_group_counts.values())
    if not has_parallel_group:
        return False

    return any(step.depends_on and step.parallel_group is not None for step in steps)


def _has_fan_out_fan_in(steps: Sequence[PlanStep]) -> bool:
    parallel_group_counts = Counter(
        step.parallel_group for step in steps if step.parallel_group is not None
    )
    if any(count > 1 for count in parallel_group_counts.values()):
        return True
    return any(len(step.depends_on) > 1 for step in steps)


def _conditional_data_quality_branch(
    plan: ExecutionPlan,
) -> _ConditionalBranch | None:
    synthesis_step = next(
        (step for step in plan.steps if step.agent_id == "synthesis"),
        None,
    )
    if synthesis_step is None:
        return None

    for step in plan.steps:
        if step.agent_id != "data_quality" or len(step.depends_on) != 1:
            continue
        if step.step_id not in synthesis_step.depends_on:
            continue
        return _ConditionalBranch(
            source_step_id=step.depends_on[0],
            data_quality_step_id=step.step_id,
            synthesis_step_id=synthesis_step.step_id,
        )

    return None


def _graph_step_ids_by_plan_step_id(steps: Sequence[PlanStep]) -> dict[str, str]:
    graph_step_ids: dict[str, str] = {}
    used_graph_step_ids: set[str] = set()
    for step in steps:
        base_id = _graph_step_id_for(step.step_id)
        graph_step_id = base_id
        suffix = 2
        while graph_step_id in used_graph_step_ids:
            graph_step_id = f"{base_id}_{suffix}"
            suffix += 1

        graph_step_ids[step.step_id] = graph_step_id
        used_graph_step_ids.add(graph_step_id)

    return graph_step_ids


def _graph_step_id_for(plan_step_id: str) -> str:
    if plan_step_id.startswith("step_"):
        suffix = plan_step_id.removeprefix("step_")
    else:
        suffix = plan_step_id
    return f"graph_step_{_identifier_suffix(suffix)}"


def _graph_id_for(plan_id: str) -> str:
    if plan_id.startswith("plan_"):
        suffix = plan_id.removeprefix("plan_")
    else:
        suffix = plan_id
    return f"graph_{_identifier_suffix(suffix)}"


def _identifier_suffix(value: str) -> str:
    identifier = re.sub(r"[^A-Za-z0-9_]+", "_", value.strip())
    identifier = identifier.strip("_")
    if not identifier:
        return "unknown"
    if identifier[0].isdigit():
        return f"id_{identifier}"
    return identifier


def _ordered_ids(ids: set[str]) -> list[str]:
    return sorted(ids)


__all__ = [
    "DEFAULT_ROUTE",
    "MISSING_INTERNAL_DATA_ROUTE",
    "BuiltGraph",
    "GraphBuildError",
    "GraphBuilder",
    "GraphPlanApprovalError",
]
