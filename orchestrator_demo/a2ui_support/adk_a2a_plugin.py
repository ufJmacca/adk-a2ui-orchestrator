"""ADK plugin that promotes A2UI tool payloads to protocol-level A2A parts."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from google.adk.a2a.converters import part_converter
from google.adk.events.event import Event
from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types as genai_types

from orchestrator_demo.a2a_support.transport import A2UI_MIME_TYPE
from orchestrator_demo.a2ui_support.adk_a2a_delivery import (
    adk_a2a_bridge_parts_for_response,
)


A2UI_A2A_TOOL_NAMES = {
    "submit_orchestrator_request",
    "add_plan_instruction",
    "remove_plan_step",
    "replace_plan_agent",
    "reorder_plan_steps",
    "approve_orchestrator_plan",
    "reject_orchestrator_plan",
}
_ORCHESTRATOR_SESSION_STATE_KEY = "orchestrator_session"


class A2uiA2AProtocolPlugin(BasePlugin):
    """Append A2A A2UI bridge parts to orchestrator tool response events."""

    def __init__(self, name: str = "a2ui_a2a_protocol_plugin") -> None:
        super().__init__(name=name)

    async def on_event_callback(
        self,
        *,
        invocation_context: Any,
        event: Event,
    ) -> Event | None:
        content = event.content
        if content is None or not content.parts:
            return None

        bridge_parts: list[genai_types.Part] = []
        existing_components_by_surface_id = _existing_components_by_surface_id(
            invocation_context
        )
        for response in _a2ui_tool_function_responses(content.parts):
            bridge_parts.extend(
                adk_a2a_bridge_parts_for_response(
                    response,
                    existing_components_by_surface_id=existing_components_by_surface_id,
                )
            )
        if not bridge_parts:
            return None

        missing_bridge_parts = _missing_a2ui_bridge_parts(
            existing_parts=content.parts,
            expected_parts=bridge_parts,
        )
        if not missing_bridge_parts:
            return None

        modified_event = event.model_copy(deep=True)
        modified_content = content.model_copy(deep=True)
        modified_content.parts = [*(content.parts or []), *missing_bridge_parts]
        modified_event.content = modified_content
        _mark_event_as_a2a_response(modified_event)
        return modified_event


def _a2ui_tool_function_responses(
    parts: Sequence[genai_types.Part],
) -> list[Mapping[str, Any]]:
    responses: list[Mapping[str, Any]] = []
    for part in parts:
        function_response = part.function_response
        if function_response is None:
            continue
        if function_response.name not in A2UI_A2A_TOOL_NAMES:
            continue
        response = function_response.response
        if isinstance(response, Mapping):
            responses.append(response)
    return responses


def _existing_components_by_surface_id(
    invocation_context: Any,
) -> Mapping[str, Mapping[str, Mapping[str, Any]]] | None:
    for state_owner in (
        getattr(invocation_context, "session", None),
        invocation_context,
    ):
        state = getattr(state_owner, "state", None)
        if not isinstance(state, Mapping):
            continue
        snapshot = state.get(_ORCHESTRATOR_SESSION_STATE_KEY)
        if not isinstance(snapshot, Mapping):
            continue
        surface_registry = snapshot.get("surfaceRegistry")
        if not isinstance(surface_registry, Mapping):
            continue
        components_by_surface_id = surface_registry.get("componentsBySurfaceId")
        if isinstance(components_by_surface_id, Mapping):
            return components_by_surface_id
    return None


def _missing_a2ui_bridge_parts(
    *,
    existing_parts: Sequence[genai_types.Part],
    expected_parts: Sequence[genai_types.Part],
) -> list[genai_types.Part]:
    existing_counts = Counter(
        _canonical_a2ui_payload(payload)
        for payload in _a2ui_payloads_from_parts(existing_parts)
    )
    seen_expected_counts: Counter[str] = Counter()
    missing_parts: list[genai_types.Part] = []
    for expected_part in expected_parts:
        payload = _a2ui_payload_from_part(expected_part)
        if payload is None:
            continue
        key = _canonical_a2ui_payload(payload)
        seen_expected_counts[key] += 1
        if seen_expected_counts[key] <= existing_counts[key]:
            continue
        missing_parts.append(expected_part)
    return missing_parts


def _a2ui_payloads_from_parts(
    parts: Sequence[genai_types.Part],
) -> list[Mapping[str, Any]]:
    payloads: list[Mapping[str, Any]] = []
    for part in parts:
        payload = _a2ui_payload_from_part(part)
        if payload is not None:
            payloads.append(payload)
    return payloads


def _a2ui_payload_from_part(part: genai_types.Part) -> Mapping[str, Any] | None:
    converted_part = part_converter.convert_genai_part_to_a2a_part(part)
    if converted_part is None:
        return None

    data_part = converted_part.root
    metadata = getattr(data_part, "metadata", None)
    data = getattr(data_part, "data", None)
    if (
        isinstance(metadata, Mapping)
        and metadata.get("mimeType") == A2UI_MIME_TYPE
        and isinstance(data, Mapping)
    ):
        return {
            "data": dict(data),
            "metadata": dict(metadata),
        }
    return None


def _canonical_a2ui_payload(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _mark_event_as_a2a_response(event: Event) -> None:
    custom_metadata = event.custom_metadata
    if isinstance(custom_metadata, Mapping):
        custom_metadata = dict(custom_metadata)
    else:
        custom_metadata = {}
    custom_metadata["a2a:response"] = True
    event.custom_metadata = custom_metadata


__all__ = [
    "A2UI_A2A_TOOL_NAMES",
    "A2uiA2AProtocolPlugin",
]
