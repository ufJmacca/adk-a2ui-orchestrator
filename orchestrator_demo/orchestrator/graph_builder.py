"""Build ADK-backed graph workflows from approved immutable plans."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
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
    AdkRuntimeStep,
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


@dataclass(frozen=True)
class _RuntimeJoinGroup:
    name: str
    target_id: str
    incoming_edges: tuple[GraphEdge, ...]


@dataclass(frozen=True)
class _ConditionalIncomingEdge:
    edge: GraphEdge
    decision_source_id: str
    route: str


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
            conditional_branch=_conditional_branch_graph_ids(
                conditional_branch,
                graph_step_ids_by_plan_step_id=graph_step_ids_by_plan_step_id,
            ),
        )
        runtime = self._runtime_factory.build(
            graph_id=spec.graph_id,
            step_node_names=[step.graph_step_id for step in spec.steps],
            step_definitions=_runtime_steps_for(
                plan,
                graph_step_ids_by_plan_step_id=graph_step_ids_by_plan_step_id,
            ),
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


def _runtime_steps_for(
    plan: ExecutionPlan,
    *,
    graph_step_ids_by_plan_step_id: dict[str, str],
) -> list[AdkRuntimeStep]:
    return [
        AdkRuntimeStep(
            node_name=graph_step_ids_by_plan_step_id[step.step_id],
            plan_id=plan.plan_id,
            plan_version=plan.plan_version,
            objective=plan.objective,
            step_id=step.step_id,
            agent_id=step.agent_id,
            instruction=step.instruction,
            expected_output=step.expected_output,
            data_source_categories=tuple(step.data_source_categories),
            depends_on=tuple(step.depends_on),
            parallel_group=step.parallel_group,
        )
        for step in plan.steps
    ]


def _graph_edges_for(
    steps: Sequence[PlanStep],
    *,
    graph_step_ids_by_plan_step_id: dict[str, str],
    conditional_branch: _ConditionalBranch | None,
) -> list[GraphEdge]:
    edges: list[GraphEdge] = []
    for step in steps:
        for dependency_id in step.depends_on:
            condition = _edge_condition_for_dependency(
                step=step,
                dependency_id=dependency_id,
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


def _edge_condition_for_dependency(
    *,
    step: PlanStep,
    dependency_id: str,
    conditional_branch: _ConditionalBranch | None,
) -> str | None:
    if conditional_branch is not None:
        branch_condition = _conditional_edge_condition(
            dependency_id=dependency_id,
            step_id=step.step_id,
            conditional_branch=conditional_branch,
        )
        if branch_condition is not None:
            return branch_condition

    if step.condition is None:
        return None
    if step.depends_on and dependency_id == step.depends_on[0]:
        return step.condition
    return None


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
    conditional_branch: _ConditionalBranch | None,
) -> tuple[list[AdkRuntimeEdge], list[str]]:
    graph_edges = list(spec.edges)
    incoming_edges_by_target = _incoming_edges_by_target(graph_edges)
    join_groups = _runtime_join_groups(
        incoming_edges_by_target,
        conditional_branch=conditional_branch,
    )
    grouped_edge_keys = {
        _edge_key(edge) for group in join_groups for edge in group.incoming_edges
    }
    join_node_names = _join_node_names_for(join_groups)

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
        if _edge_key(edge) in grouped_edge_keys:
            continue
        _append_runtime_edge(
            runtime_edges,
            AdkRuntimeEdge(
                from_node_name=edge.from_step_id,
                to_node_name=edge.to_step_id,
                route=edge.condition,
            ),
        )

    for group in join_groups:
        for edge in group.incoming_edges:
            if _requires_route_gate(group, edge):
                gate_name = _route_gate_join_name(group, edge)
                _append_runtime_edge(
                    runtime_edges,
                    AdkRuntimeEdge(
                        from_node_name=edge.from_step_id,
                        to_node_name=gate_name,
                        route=edge.condition,
                    ),
                )
                _append_runtime_edge(
                    runtime_edges,
                    AdkRuntimeEdge(
                        from_node_name=gate_name,
                        to_node_name=group.name,
                    ),
                )
            else:
                _append_runtime_edge(
                    runtime_edges,
                    AdkRuntimeEdge(
                        from_node_name=edge.from_step_id,
                        to_node_name=group.name,
                        route=edge.condition,
                    ),
                )
        _append_runtime_edge(
            runtime_edges,
            AdkRuntimeEdge(
                from_node_name=group.name,
                to_node_name=group.target_id,
            ),
        )

    return runtime_edges, join_node_names


def _join_node_names_for(join_groups: Sequence[_RuntimeJoinGroup]) -> list[str]:
    join_node_names: list[str] = []
    for group in join_groups:
        join_node_names.append(group.name)
        for edge in group.incoming_edges:
            if _requires_route_gate(group, edge):
                join_node_names.append(_route_gate_join_name(group, edge))
    return join_node_names


def _requires_route_gate(group: _RuntimeJoinGroup, edge: GraphEdge) -> bool:
    return edge.condition is not None and len(group.incoming_edges) > 1


def _route_gate_join_name(group: _RuntimeJoinGroup, edge: GraphEdge) -> str:
    assert edge.condition is not None
    route_suffix = _route_suffix(edge.condition)
    if group.name.endswith(f"_{route_suffix}"):
        return f"{group.name}_gate"
    return f"{group.name}_{route_suffix}_gate"


def _runtime_join_groups(
    incoming_edges_by_target: dict[str, list[GraphEdge]],
    *,
    conditional_branch: _ConditionalBranch | None,
) -> list[_RuntimeJoinGroup]:
    conditional_join_groups = _conditional_runtime_join_groups(
        incoming_edges_by_target,
        conditional_branch=conditional_branch,
    )
    conditional_join_targets = {group.target_id for group in conditional_join_groups}
    if conditional_branch is not None:
        conditional_join_targets.add(conditional_branch.synthesis_step_id)

    join_groups = list(conditional_join_groups)
    for target_id in _ordered_ids(
        set(incoming_edges_by_target) - conditional_join_targets
    ):
        incoming_edges = incoming_edges_by_target[target_id]
        generic_join_groups, is_conditional_target = (
            _generic_conditional_runtime_join_groups(
                target_id=target_id,
                incoming_edges=incoming_edges,
                incoming_edges_by_target=incoming_edges_by_target,
            )
        )
        if is_conditional_target:
            join_groups.extend(generic_join_groups)
            continue
        if len(incoming_edges) <= 1:
            continue
        join_groups.append(
            _RuntimeJoinGroup(
                name=f"join_{target_id}",
                target_id=target_id,
                incoming_edges=tuple(incoming_edges),
            )
        )

    return join_groups


def _generic_conditional_runtime_join_groups(
    *,
    target_id: str,
    incoming_edges: Sequence[GraphEdge],
    incoming_edges_by_target: dict[str, list[GraphEdge]],
) -> tuple[list[_RuntimeJoinGroup], bool]:
    conditionals = [
        conditional
        for edge in incoming_edges
        if (
            conditional := _conditional_incoming_edge(
                edge,
                incoming_edges_by_target=incoming_edges_by_target,
            )
        )
        is not None
    ]
    if not conditionals:
        return [], False

    conditional_edge_keys = {_edge_key(conditional.edge) for conditional in conditionals}
    mandatory_edges = [
        edge for edge in incoming_edges if _edge_key(edge) not in conditional_edge_keys
    ]
    route_edges_by_source = _route_edges_by_source(conditionals)
    route_groups_by_source = [
        tuple(route_edges.items()) for route_edges in route_edges_by_source.values()
    ]
    needs_join = (
        bool(mandatory_edges)
        or len(route_groups_by_source) > 1
        or any(
            len(edges) > 1
            for route_edges in route_groups_by_source
            for _, edges in route_edges
        )
    )
    if not needs_join:
        return [], True

    route_combinations = list(product(*route_groups_by_source))
    use_unsuffixed_join_name = len(route_combinations) == 1
    base_join_names = [
        _conditional_join_name(
            target_id,
            routes=tuple(route for route, _ in route_combination),
            use_unsuffixed_name=use_unsuffixed_join_name,
        )
        for route_combination in route_combinations
    ]
    duplicate_join_names = {
        name for name, count in Counter(base_join_names).items() if count > 1
    }
    duplicate_join_name_ordinals: Counter[str] = Counter()
    used_join_names: set[str] = set()
    join_groups: list[_RuntimeJoinGroup] = []
    for route_combination, base_join_name in zip(
        route_combinations,
        base_join_names,
    ):
        branch_edges = [
            edge
            for _, route_edges in route_combination
            for edge in route_edges
        ]
        join_name = base_join_name
        if base_join_name in duplicate_join_names:
            while True:
                duplicate_join_name_ordinals[base_join_name] += 1
                join_name = (
                    f"{base_join_name}_route"
                    f"{duplicate_join_name_ordinals[base_join_name]}"
                )
                if join_name not in used_join_names:
                    break
        elif join_name in used_join_names:
            ordinal = 2
            while f"{base_join_name}_{ordinal}" in used_join_names:
                ordinal += 1
            join_name = f"{base_join_name}_{ordinal}"
        used_join_names.add(join_name)
        join_groups.append(
            _RuntimeJoinGroup(
                name=join_name,
                target_id=target_id,
                incoming_edges=tuple([*branch_edges, *mandatory_edges]),
            )
        )

    return join_groups, True


def _conditional_incoming_edge(
    edge: GraphEdge,
    *,
    incoming_edges_by_target: dict[str, list[GraphEdge]],
) -> _ConditionalIncomingEdge | None:
    if edge.condition is not None:
        return _ConditionalIncomingEdge(
            edge=edge,
            decision_source_id=edge.from_step_id,
            route=edge.condition,
        )

    predecessor_route_edges = [
        predecessor_edge
        for predecessor_edge in incoming_edges_by_target.get(edge.from_step_id, [])
        if predecessor_edge.condition is not None
    ]
    if len(predecessor_route_edges) != 1:
        return None

    route_edge = predecessor_route_edges[0]
    route = route_edge.condition
    if route is None:
        return None
    return _ConditionalIncomingEdge(
        edge=edge,
        decision_source_id=route_edge.from_step_id,
        route=route,
    )


def _route_edges_by_source(
    conditionals: Sequence[_ConditionalIncomingEdge],
) -> dict[str, dict[str, tuple[GraphEdge, ...]]]:
    route_edges_by_source: dict[str, dict[str, list[GraphEdge]]] = {}
    for conditional in conditionals:
        source_routes = route_edges_by_source.setdefault(
            conditional.decision_source_id,
            {},
        )
        source_routes.setdefault(conditional.route, []).append(conditional.edge)

    return {
        source_id: {
            route: tuple(edges)
            for route, edges in route_edges.items()
        }
        for source_id, route_edges in route_edges_by_source.items()
    }


def _conditional_join_name(
    target_id: str,
    *,
    routes: Sequence[str],
    use_unsuffixed_name: bool,
) -> str:
    if use_unsuffixed_name:
        return f"join_{target_id}"
    route_suffix = "_".join(_route_suffix(route) for route in routes)
    return f"join_{target_id}_{route_suffix}"


def _conditional_runtime_join_groups(
    incoming_edges_by_target: dict[str, list[GraphEdge]],
    *,
    conditional_branch: _ConditionalBranch | None,
) -> list[_RuntimeJoinGroup]:
    if conditional_branch is None:
        return []

    target_id = conditional_branch.synthesis_step_id
    incoming_edges = incoming_edges_by_target.get(target_id, [])
    if len(incoming_edges) <= 2:
        return []

    join_groups, is_conditional_target = _generic_conditional_runtime_join_groups(
        target_id=target_id,
        incoming_edges=incoming_edges,
        incoming_edges_by_target=incoming_edges_by_target,
    )
    if not is_conditional_target:
        return []
    return join_groups


def _edge_key(edge: GraphEdge) -> tuple[str, str, str | None]:
    return (edge.from_step_id, edge.to_step_id, edge.condition)


def _route_suffix(route: str) -> str:
    if route == DEFAULT_ROUTE:
        return "default"
    return _identifier_suffix(str(route))


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
    if any(step.condition is not None for step in plan.steps):
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
        if (
            step.agent_id != "data_quality"
            or step.condition != MISSING_INTERNAL_DATA_ROUTE
            or len(step.depends_on) != 1
        ):
            continue
        if step.step_id not in synthesis_step.depends_on:
            continue
        return _ConditionalBranch(
            source_step_id=step.depends_on[0],
            data_quality_step_id=step.step_id,
            synthesis_step_id=synthesis_step.step_id,
        )

    return None


def _conditional_branch_graph_ids(
    conditional_branch: _ConditionalBranch | None,
    *,
    graph_step_ids_by_plan_step_id: dict[str, str],
) -> _ConditionalBranch | None:
    if conditional_branch is None:
        return None

    return _ConditionalBranch(
        source_step_id=graph_step_ids_by_plan_step_id[
            conditional_branch.source_step_id
        ],
        data_quality_step_id=graph_step_ids_by_plan_step_id[
            conditional_branch.data_quality_step_id
        ],
        synthesis_step_id=graph_step_ids_by_plan_step_id[
            conditional_branch.synthesis_step_id
        ],
    )


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
