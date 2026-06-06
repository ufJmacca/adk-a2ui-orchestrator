"""Contract-aware redaction for outward tool response payloads."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from orchestrator_demo.a2ui_support.secret_safety import (
    REDACTED_SECRET,
    is_secret_like_field_name,
    redact_secret_values_only,
    safe_path_component,
)


_CONTRACT_VALUE_KEYS = {
    "child",
    "children",
    "code",
    "component_id",
    "data_source_categories",
    "depends_on",
    "detected_intents",
    "edge_route",
    "edge_routes",
    "graph_route",
    "graph_routes",
    "owner_type",
    "parallel_group",
    "parallel_groups",
    "path",
    "required_agents",
    "required_fields",
    "condition",
    "conditions",
    "route",
    "route_code",
    "route_condition",
    "route_conditions",
    "routes",
    "selected_agent",
    "selected_agents",
    "selected_condition",
    "selected_conditions",
    "selected_route",
    "selected_routes",
    "status",
    "tool_name",
    "type",
}
_A2UI_COMPONENT_REFERENCE_VALUE_KEYS = {"content", "trigger"}


def redacted_response_json_safe(
    value: Any,
    *,
    parent_key: str | None = None,
    parent_is_a2ui_component: bool = False,
) -> Any:
    """Return JSON-safe response data without corrupting contract identifiers."""

    if isinstance(value, Mapping):
        is_a2ui_component = _is_a2ui_component(value)
        return {
            safe_path_component(str(key)): (
                REDACTED_SECRET
                if is_secret_like_field_name(str(key))
                else redacted_response_json_safe(
                    child,
                    parent_key=str(key),
                    parent_is_a2ui_component=is_a2ui_component,
                )
            )
            for key, child in value.items()
        }
    if isinstance(value, tuple | list):
        return [
            redacted_response_json_safe(
                child,
                parent_key=parent_key,
                parent_is_a2ui_component=parent_is_a2ui_component,
            )
            for child in value
        ]
    if isinstance(value, set | frozenset):
        return sorted(
            redacted_response_json_safe(
                child,
                parent_key=parent_key,
                parent_is_a2ui_component=parent_is_a2ui_component,
            )
            for child in value
        )
    if isinstance(value, str):
        if _is_contract_value_key(
            parent_key,
            parent_is_a2ui_component=parent_is_a2ui_component,
        ):
            return redact_secret_values_only(value)
        return redact_secret_values_only(value)
    return value


def _is_contract_value_key(
    key: str | None,
    *,
    parent_is_a2ui_component: bool,
) -> bool:
    if key is None:
        return False
    normalized = _maybe_camel_case_to_snake(key)
    if parent_is_a2ui_component and normalized in _A2UI_COMPONENT_REFERENCE_VALUE_KEYS:
        return True
    return (
        normalized == "id"
        or normalized.endswith("_id")
        or normalized.endswith("_ids")
        or normalized in _CONTRACT_VALUE_KEYS
    )


def _maybe_camel_case_to_snake(value: str) -> str:
    normalized = "".join(
        f"_{char.lower()}" if char.isupper() else char for char in value
    )
    return normalized.strip("_").replace("-", "_").lower()


def _is_a2ui_component(value: Mapping[Any, Any]) -> bool:
    return isinstance(value.get("id"), str) and isinstance(
        value.get("component", value.get("type")),
        str,
    )


__all__ = ["redacted_response_json_safe"]
