"""Shared Pydantic contracts for the orchestrator demo.

The models in this module are intentionally side-effect free so tests and later
runtime modules can import them without requiring model-provider secrets.
"""

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator


IntentName = Literal[
    "industry_research",
    "web_search",
    "internal_knowledge",
    "meeting_prep",
    "prospect_research",
    "credit_risk",
    "relationship_summary",
    "product_opportunity",
    "compliance_policy",
    "data_quality",
    "unknown",
]
Complexity = Literal["simple", "complex"]
RoutingPath = Literal["direct", "plan_required", "clarification_required"]
ExecutionMode = Literal["local_llm", "local_a2a_compatible", "remote_a2a"]
GraphPattern = Literal[
    "direct",
    "sequential",
    "fan_out_fan_in",
    "conditional",
    "mixed",
]
UserActionType = Literal[
    "approve_plan",
    "reject_plan",
    "edit_plan",
    "remove_step",
    "reorder_steps",
    "choose_agent",
    "replace_agent",
    "add_instruction",
    "add_instructions",
    "specialist_action",
]
A2uiPayload = dict[str, Any] | list[dict[str, Any]]
PLAN_USER_ACTION_TYPES: set[str] = {
    "approve_plan",
    "reject_plan",
    "edit_plan",
    "remove_step",
    "reorder_steps",
    "choose_agent",
    "replace_agent",
    "add_instruction",
    "add_instructions",
}
PLAN_APPROVAL_SURFACE_PREFIX = "surface_plan_"
PLAN_APPROVAL_SURFACE_PATTERN = (
    rf"^{PLAN_APPROVAL_SURFACE_PREFIX}[A-Za-z0-9][A-Za-z0-9_-]*$"
)
StatusName = Literal[
    "plan_approved",
    "graph_created",
    "step_started",
    "step_completed",
    "step_failed",
    "parallel_branch_started",
    "parallel_branch_completed",
    "synthesis_started",
    "final_response_ready",
]


def _duplicate_values(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []

    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)

    return duplicates


def _find_dependency_cycle(dependencies_by_id: dict[str, list[str]]) -> list[str] | None:
    visiting: set[str] = set()
    visited: set[str] = set()
    path: list[str] = []

    def visit(node_id: str) -> list[str] | None:
        if node_id in visited:
            return None
        if node_id in visiting:
            cycle_start = path.index(node_id)
            return [*path[cycle_start:], node_id]

        visiting.add(node_id)
        path.append(node_id)
        for dependency_id in dependencies_by_id[node_id]:
            cycle = visit(dependency_id)
            if cycle is not None:
                return cycle
        path.pop()
        visiting.remove(node_id)
        visited.add(node_id)

        return None

    for node_id in dependencies_by_id:
        cycle = visit(node_id)
        if cycle is not None:
            return cycle

    return None


def _is_plan_scoped_user_action(action_type: Any, surface_id: Any) -> bool:
    return (
        isinstance(action_type, str)
        and action_type in PLAN_USER_ACTION_TYPES
        and _is_plan_approval_surface(surface_id)
    )


def _is_plan_approval_surface(surface_id: Any) -> bool:
    return isinstance(surface_id, str) and surface_id.startswith(
        PLAN_APPROVAL_SURFACE_PREFIX
    )


def _validate_dependency_topology(
    *,
    contract_name: str,
    id_field_name: str,
    declared_ids: list[str],
    dependencies_by_id: dict[str, list[str]],
    reference_source_label: str,
    declared_reference_label: str,
) -> None:
    duplicate_ids = _duplicate_values(declared_ids)
    if duplicate_ids:
        duplicates = ", ".join(duplicate_ids)
        raise ValueError(
            f"{contract_name} {id_field_name} values must be unique: {duplicates}"
        )

    duplicate_dependencies_by_id = {
        node_id: duplicate_dependencies
        for node_id, dependency_ids in dependencies_by_id.items()
        if (duplicate_dependencies := _duplicate_values(dependency_ids))
    }
    if duplicate_dependencies_by_id:
        duplicates = "; ".join(
            f"{node_id}: {', '.join(duplicate_dependencies)}"
            for node_id, duplicate_dependencies in sorted(
                duplicate_dependencies_by_id.items()
            )
        )
        raise ValueError(
            f"{contract_name} {reference_source_label} entries must be unique "
            f"per {id_field_name}: {duplicates}"
        )

    declared_id_set = set(declared_ids)
    undeclared_ids = {
        node_id
        for node_id in dependencies_by_id
        if node_id not in declared_id_set
    }
    undeclared_ids.update(
        dependency_id
        for dependency_ids in dependencies_by_id.values()
        for dependency_id in dependency_ids
        if dependency_id not in declared_id_set
    )
    if undeclared_ids:
        missing = ", ".join(sorted(undeclared_ids))
        raise ValueError(
            f"{contract_name} {reference_source_label} must reference "
            f"{declared_reference_label}: {missing}"
        )

    self_references = {
        node_id
        for node_id, dependency_ids in dependencies_by_id.items()
        if node_id in dependency_ids
    }
    if self_references:
        invalid = ", ".join(sorted(self_references))
        raise ValueError(
            f"{contract_name} {reference_source_label} cannot reference "
            f"themselves: {invalid}"
        )

    cycle = _find_dependency_cycle(dependencies_by_id)
    if cycle is not None:
        cycle_path = " -> ".join(cycle)
        raise ValueError(
            f"{contract_name} {reference_source_label} must be acyclic: "
            f"{cycle_path}"
        )


class ContractModel(BaseModel):
    """Base class for strict shared data contracts."""

    model_config = ConfigDict(extra="forbid")


class IntentSuggestion(ContractModel):
    intent: IntentName
    confidence: float = Field(ge=0.0, le=1.0)


class LlmIntentAssessment(ContractModel):
    intents: list[IntentName] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    complexity: Complexity
    rationale: str = Field(min_length=1)
    required_agents: list[str] = Field(min_length=1)


class RoutingDecision(ContractModel):
    path: RoutingPath
    selected_agent: str | None
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_agent_for_direct_path(self) -> "RoutingDecision":
        if self.path == "direct" and (
            self.selected_agent is None or not self.selected_agent.strip()
        ):
            raise ValueError("direct routing decisions require selected_agent")

        return self


class PlanStep(ContractModel):
    step_id: str = Field(pattern=r"^step_[A-Za-z0-9][A-Za-z0-9_-]*$")
    agent_id: str = Field(min_length=1)
    instruction: str = Field(min_length=1)
    depends_on: list[str] = Field(default_factory=list)
    condition: str | None = None
    expected_output: str = Field(min_length=1)
    data_source_categories: list[str] = Field(default_factory=list)
    parallel_group: str | None = None


class ExecutionPlan(ContractModel):
    plan_id: str = Field(pattern=r"^plan_[A-Za-z0-9][A-Za-z0-9_-]*$")
    objective: str = Field(min_length=1)
    detected_intents: list[IntentName] = Field(default_factory=list)
    selected_agents: list[str] = Field(default_factory=list)
    steps: list[PlanStep] = Field(min_length=1)
    data_source_categories: list[str] = Field(default_factory=list)
    risk_notes: list[str] = Field(default_factory=list)
    approval_surface_id: str | None = Field(
        default=None,
        pattern=PLAN_APPROVAL_SURFACE_PATTERN,
    )
    plan_version: int = Field(default=1, ge=1)
    immutable_after_approval: bool = True

    @model_validator(mode="after")
    def validate_step_dependencies(self) -> "ExecutionPlan":
        declared_step_ids = [step.step_id for step in self.steps]
        dependencies_by_step_id = {
            step.step_id: step.depends_on for step in self.steps
        }
        _validate_dependency_topology(
            contract_name="ExecutionPlan",
            id_field_name="step_id",
            declared_ids=declared_step_ids,
            dependencies_by_id=dependencies_by_step_id,
            reference_source_label="dependencies",
            declared_reference_label="declared plan steps",
        )

        return self


class AgentDescriptor(ContractModel):
    agent_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    capabilities: list[str]
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    a2ui_catalogs: list[str]
    routing_examples: list[str]
    execution_mode: ExecutionMode


class SpecialistRequest(ContractModel):
    request_id: str = Field(pattern=r"^request_[A-Za-z0-9][A-Za-z0-9_-]*$")
    user_input: str = Field(min_length=1)
    agent_id: str = Field(min_length=1)
    plan_id: str | None = Field(
        default=None,
        pattern=r"^plan_[A-Za-z0-9][A-Za-z0-9_-]*$",
    )
    step_id: str | None = Field(
        default=None,
        pattern=r"^step_[A-Za-z0-9][A-Za-z0-9_-]*$",
    )
    context: dict[str, Any] = Field(default_factory=dict)


class SpecialistResponse(ContractModel):
    response_id: str = Field(pattern=r"^response_[A-Za-z0-9][A-Za-z0-9_-]*$")
    agent_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    structured_output: dict[str, Any] = Field(default_factory=dict)
    a2ui_payload: A2uiPayload | None = None
    surface_id: str | None = Field(
        default=None,
        pattern=r"^surface_[A-Za-z0-9][A-Za-z0-9_-]*$",
    )


class GraphStep(ContractModel):
    graph_step_id: str = Field(pattern=r"^graph_step_[A-Za-z0-9][A-Za-z0-9_-]*$")
    plan_step_id: str = Field(pattern=r"^step_[A-Za-z0-9][A-Za-z0-9_-]*$")
    agent_id: str = Field(min_length=1)
    depends_on: list[str] = Field(default_factory=list)
    parallel_group: str | None = None


class GraphEdge(ContractModel):
    from_step_id: str = Field(pattern=r"^graph_step_[A-Za-z0-9][A-Za-z0-9_-]*$")
    to_step_id: str = Field(pattern=r"^graph_step_[A-Za-z0-9][A-Za-z0-9_-]*$")
    condition: str | None = None


class GraphSpec(ContractModel):
    graph_id: str = Field(pattern=r"^graph_[A-Za-z0-9][A-Za-z0-9_-]*$")
    plan_id: str = Field(pattern=r"^plan_[A-Za-z0-9][A-Za-z0-9_-]*$")
    pattern: GraphPattern
    steps: list[GraphStep] = Field(min_length=1)
    edges: list[GraphEdge] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_edge_endpoints(self) -> "GraphSpec":
        declared_step_ids = [step.graph_step_id for step in self.steps]
        dependencies_by_step_id = {
            step.graph_step_id: list(step.depends_on) for step in self.steps
        }
        duplicate_edges = _duplicate_values(
            [
                f"{edge.from_step_id}->{edge.to_step_id}"
                for edge in self.edges
            ]
        )
        if duplicate_edges:
            duplicates = ", ".join(duplicate_edges)
            raise ValueError(f"GraphSpec edges must be unique: {duplicates}")

        for edge in self.edges:
            dependencies = dependencies_by_step_id.setdefault(edge.to_step_id, [])
            if edge.from_step_id not in dependencies:
                dependencies.append(edge.from_step_id)

        _validate_dependency_topology(
            contract_name="GraphSpec",
            id_field_name="graph_step_id",
            declared_ids=declared_step_ids,
            dependencies_by_id=dependencies_by_step_id,
            reference_source_label="edges and dependencies",
            declared_reference_label="declared graph steps",
        )

        return self


class UserAction(ContractModel):
    action_id: str | None = Field(
        default=None,
        pattern=r"^action_[A-Za-z0-9][A-Za-z0-9_-]*$",
        validation_alias=AliasChoices("actionId", "action_id"),
        serialization_alias="actionId",
    )
    type: str = Field(min_length=1)
    surface_id: str = Field(
        pattern=r"^surface_[A-Za-z0-9][A-Za-z0-9_-]*$",
        validation_alias=AliasChoices("surfaceId", "surface_id"),
        serialization_alias="surfaceId",
    )
    plan_id: str | None = Field(
        default=None,
        pattern=r"^plan_[A-Za-z0-9][A-Za-z0-9_-]*$",
        validation_alias=AliasChoices("planId", "plan_id"),
        serialization_alias="planId",
    )
    plan_version: int | None = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices("planVersion", "plan_version"),
        serialization_alias="planVersion",
    )
    payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def normalize_payload_plan_identifiers(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        user_action = data.get("userAction")
        if isinstance(user_action, dict):
            data = user_action

        payload = data.get("payload")
        if not isinstance(payload, dict):
            return data

        action_type = data.get("type")
        surface_id = data.get("surfaceId", data.get("surface_id"))
        if not _is_plan_scoped_user_action(action_type, surface_id):
            return data

        normalized = dict(data)
        top_level_plan_ids = [
            value
            for value in (normalized.get("planId"), normalized.get("plan_id"))
            if value is not None
        ]
        if top_level_plan_ids and any(
            top_level_plan_id != top_level_plan_ids[0]
            for top_level_plan_id in top_level_plan_ids[1:]
        ):
            raise ValueError("top-level planId aliases must match")

        payload_plan_ids = [
            value
            for value in (payload.get("planId"), payload.get("plan_id"))
            if value is not None
        ]
        if payload_plan_ids and any(
            payload_plan_id != payload_plan_ids[0]
            for payload_plan_id in payload_plan_ids[1:]
        ):
            raise ValueError("payload planId aliases must match")

        top_level_plan_id = top_level_plan_ids[0] if top_level_plan_ids else None
        payload_plan_id = payload_plan_ids[0] if payload_plan_ids else None
        if (
            top_level_plan_id is not None
            and payload_plan_id is not None
            and top_level_plan_id != payload_plan_id
        ):
            raise ValueError("top-level planId must match payload planId")
        if top_level_plan_id is None and payload_plan_id is not None:
            normalized["planId"] = payload_plan_id

        top_level_plan_version = normalized.get("planVersion", normalized.get("plan_version"))
        payload_plan_versions = [
            value
            for value in (
                payload.get("planVersion"),
                payload.get("plan_version"),
                payload.get("editedPlanVersion"),
                payload.get("edited_plan_version"),
            )
            if value is not None
        ]
        if payload_plan_versions and any(
            payload_plan_version != payload_plan_versions[0]
            for payload_plan_version in payload_plan_versions[1:]
        ):
            raise ValueError("payload planVersion aliases must match")
        if top_level_plan_version is not None:
            for payload_plan_version in payload_plan_versions:
                if top_level_plan_version != payload_plan_version:
                    raise ValueError("top-level planVersion must match payload planVersion")
        elif payload_plan_versions:
            normalized["planVersion"] = payload_plan_versions[0]

        return normalized

    @model_validator(mode="after")
    def require_plan_identifiers_for_plan_actions(self) -> "UserAction":
        if (
            _is_plan_approval_surface(self.surface_id)
            and self.type not in PLAN_USER_ACTION_TYPES
        ):
            raise ValueError("plan approval surfaces require plan user action types")

        if _is_plan_scoped_user_action(self.type, self.surface_id):
            missing_fields = []
            if self.plan_id is None:
                missing_fields.append("planId")
            if self.type != "reject_plan" and self.plan_version is None:
                missing_fields.append("planVersion")
            if missing_fields:
                required_fields = " and ".join(missing_fields)
                raise ValueError(f"plan user actions require {required_fields}")

        return self


class StatusEvent(ContractModel):
    event_id: str = Field(pattern=r"^event_[A-Za-z0-9][A-Za-z0-9_-]*$")
    graph_id: str | None = Field(
        default=None,
        pattern=r"^graph_[A-Za-z0-9][A-Za-z0-9_-]*$",
    )
    plan_id: str | None = Field(
        default=None,
        pattern=r"^plan_[A-Za-z0-9][A-Za-z0-9_-]*$",
    )
    step_id: str | None = Field(
        default=None,
        pattern=r"^graph_step_[A-Za-z0-9][A-Za-z0-9_-]*$",
    )
    status: StatusName
    message: str = Field(min_length=1)
    details: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "A2uiPayload",
    "AgentDescriptor",
    "Complexity",
    "ContractModel",
    "ExecutionMode",
    "ExecutionPlan",
    "GraphEdge",
    "GraphPattern",
    "GraphSpec",
    "GraphStep",
    "IntentName",
    "IntentSuggestion",
    "LlmIntentAssessment",
    "PlanStep",
    "RoutingDecision",
    "RoutingPath",
    "SpecialistRequest",
    "SpecialistResponse",
    "StatusEvent",
    "StatusName",
    "UserAction",
    "UserActionType",
]
