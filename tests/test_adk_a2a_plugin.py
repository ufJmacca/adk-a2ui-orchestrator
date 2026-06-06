from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from a2a import types as a2a_types
from google.adk.a2a.converters.from_adk_event import convert_event_to_a2a_events
from google.adk.events.event import Event
from google.genai import types as genai_types

from orchestrator_demo.a2ui_support.schema_manager import A2UI_VERSION


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
A2UI_MIME_TYPE = "application/json+a2ui"
EXPECTED_TOOL_NAMES = {
    "submit_orchestrator_request",
    "add_plan_instruction",
    "remove_plan_step",
    "replace_plan_agent",
    "reorder_plan_steps",
    "approve_orchestrator_plan",
    "reject_orchestrator_plan",
}
FORBIDDEN_RENDERER_MARKERS = (
    "/api/request",
    "/api/user-action",
    "/api/status",
    "/api/status/stream",
    "/api/artifacts",
    "/static/renderer.js",
)


def _a2ui_part(surface_id: str, text: str) -> dict[str, Any]:
    return {
        "type": "data",
        "data": {
            "version": A2UI_VERSION,
            "updateComponents": {
                "surfaceId": surface_id,
                "components": [
                    {
                        "component": "Text",
                        "id": "root",
                        "text": text,
                    }
                ],
            },
        },
        "metadata": {"mimeType": A2UI_MIME_TYPE},
    }


def _incremental_a2ui_part(surface_id: str) -> dict[str, Any]:
    return {
        "type": "data",
        "data": {
            "version": A2UI_VERSION,
            "updateComponents": {
                "surfaceId": surface_id,
                "components": [
                    {
                        "component": "Table",
                        "id": "component_table",
                        "columns": [{"key": "name", "label": "Name"}],
                        "rows": [{"name": "ABC Manufacturing"}],
                    },
                    {
                        "component": "Column",
                        "id": "root",
                        "children": ["component_existing_summary"],
                    },
                ],
            },
        },
        "metadata": {"mimeType": A2UI_MIME_TYPE},
    }


def _invocation_context_with_existing_component(surface_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        session=SimpleNamespace(
            state={
                "orchestrator_session": {
                    "surfaceRegistry": {
                        "componentsBySurfaceId": {
                            surface_id: {
                                "component_existing_summary": {
                                    "component": "Text",
                                    "id": "component_existing_summary",
                                    "text": "Existing summary.",
                                }
                            }
                        }
                    }
                }
            }
        )
    )


def _function_response_event(
    tool_name: str,
    response: dict[str, Any],
) -> Event:
    function_response_part = genai_types.Part.from_function_response(
        name=tool_name,
        response=response,
    )
    assert function_response_part.function_response is not None
    function_response_part.function_response.id = f"call_{tool_name}"
    return Event(
        invocation_id="invocation_a2ui_plugin",
        author="orchestrator",
        content=genai_types.Content(role="user", parts=[function_response_part]),
    )


def _a2a_data_parts_from_event(event: Event) -> list[a2a_types.DataPart]:
    a2a_events = convert_event_to_a2a_events(
        event,
        {},
        "task_a2ui_plugin",
        "context_a2ui_plugin",
    )
    return [
        part.root
        for a2a_event in a2a_events
        if getattr(a2a_event, "artifact", None) is not None
        for part in a2a_event.artifact.parts
        if isinstance(part.root, a2a_types.DataPart)
    ]


def _a2ui_data_parts_from_event(event: Event) -> list[a2a_types.DataPart]:
    return [
        part
        for part in _a2a_data_parts_from_event(event)
        if isinstance(part.metadata, dict)
        and part.metadata.get("mimeType") == A2UI_MIME_TYPE
    ]


def _assert_no_custom_renderer_messages(payload: Any) -> None:
    serialized = json.dumps(payload, sort_keys=True)
    for marker in FORBIDDEN_RENDERER_MARKERS:
        assert marker not in serialized


def test_a2a_converter_captured_before_agent_import_exports_standard_a2ui() -> None:
    script = r"""
import json

from a2a import types as a2a_types
from google.adk.a2a.executor.config import A2aAgentExecutorConfig
from google.adk.events.event import Event
from google.genai import types as genai_types

from orchestrator_demo.a2ui_support.adk_ui_delivery import (
    adk_dev_ui_content_parts_for_a2ui_response,
)
from orchestrator_demo.a2ui_support.schema_manager import A2UI_VERSION


A2UI_MIME_TYPE = "application/json+a2ui"
standard_data = {
    "version": A2UI_VERSION,
    "updateComponents": {
        "surfaceId": "surface_stale_executor",
        "components": [
            {
                "component": "Text",
                "id": "root",
                "text": "Review this draft plan.",
            }
        ],
    },
}
response = {
    "status": "plan_required",
    "approvalSurfaceId": "surface_stale_executor",
    "a2uiParts": [
        {
            "type": "data",
            "data": standard_data,
            "metadata": {"mimeType": A2UI_MIME_TYPE},
        }
    ],
}
function_response_part = genai_types.Part.from_function_response(
    name="submit_orchestrator_request",
    response=response,
)
event = Event(
    invocation_id="invocation_stale_executor",
    author="orchestrator",
    content=genai_types.Content(
        role="user",
        parts=[
            function_response_part,
            *adk_dev_ui_content_parts_for_a2ui_response(response),
        ],
    ),
    custom_metadata={"a2a:response": True},
)


def exported_a2ui_data(converter):
    a2a_events = converter(
        event,
        {},
        "task_stale_executor",
        "context_stale_executor",
    )
    return [
        part.root.data
        for a2a_event in a2a_events
        if getattr(a2a_event, "artifact", None) is not None
        for part in a2a_event.artifact.parts
        if isinstance(part.root, a2a_types.DataPart)
        and isinstance(part.root.metadata, dict)
        and part.root.metadata.get("mimeType") == A2UI_MIME_TYPE
    ]


config = A2aAgentExecutorConfig()
captured_converter = config.adk_event_converter
before_import = exported_a2ui_data(captured_converter)

import orchestrator_demo.orchestrator.agent  # noqa: F401

after_import = exported_a2ui_data(captured_converter)
print(
    json.dumps(
        {
            "before": before_import,
            "after": after_import,
            "expected": standard_data,
        },
        sort_keys=True,
    )
)
"""

    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])

    assert len(payload["before"]) == 1
    assert "surfaceUpdate" in payload["before"][0]
    assert "updateComponents" not in payload["before"][0]
    assert payload["after"] == [payload["expected"]]


@pytest.mark.asyncio
async def test_a2ui_a2a_protocol_plugin_appends_top_level_data_parts_once() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.adk_a2a_plugin import (
        A2UI_A2A_TOOL_NAMES,
        A2uiA2AProtocolPlugin,
    )

    response = {
        "status": "plan_required",
        "path": "plan_required",
        "planId": "plan_bridge",
        "planVersion": 1,
        "approvalSurfaceId": "surface_bridge_plan",
        "a2uiParts": [
            _a2ui_part("surface_bridge_plan", "Review this draft plan."),
            _a2ui_part("surface_bridge_specialist", "Specialist surface."),
        ],
    }
    event = _function_response_event("submit_orchestrator_request", response)
    plugin = A2uiA2AProtocolPlugin()

    # Act
    modified_event = await plugin.on_event_callback(
        invocation_context=SimpleNamespace(),
        event=event,
    )
    assert modified_event is not None
    second_pass = await plugin.on_event_callback(
        invocation_context=SimpleNamespace(),
        event=modified_event,
    )
    event_after_second_pass = second_pass or modified_event

    # Assert
    assert A2UI_A2A_TOOL_NAMES == EXPECTED_TOOL_NAMES
    assert modified_event is not event
    assert modified_event.content is not None
    assert modified_event.content.parts is not None
    assert len(modified_event.content.parts) == 3

    function_response = modified_event.content.parts[0].function_response
    assert function_response is not None
    assert function_response.name == "submit_orchestrator_request"
    assert function_response.response == response
    assert modified_event.content.parts[0].inline_data is None

    assert event_after_second_pass.content is not None
    assert event_after_second_pass.content.parts is not None
    assert len(event_after_second_pass.content.parts) == 3

    a2ui_data_parts = _a2ui_data_parts_from_event(event_after_second_pass)
    assert [part.data for part in a2ui_data_parts] == [
        part["data"] for part in response["a2uiParts"]
    ]
    assert all(
        part.metadata == {"mimeType": A2UI_MIME_TYPE} for part in a2ui_data_parts
    )

    fallback_response_parts = [
        part
        for part in _a2a_data_parts_from_event(event_after_second_pass)
        if part.metadata
        and part.metadata.get("adk_type") == "function_response"
        and part.data.get("name") == "submit_orchestrator_request"
    ]
    assert len(fallback_response_parts) == 1
    assert fallback_response_parts[0].data["response"] == response

    dumped_event = event_after_second_pass.model_dump(
        by_alias=True,
        mode="json",
        exclude_none=True,
    )
    _assert_no_custom_renderer_messages(dumped_event)


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", sorted(EXPECTED_TOOL_NAMES))
async def test_a2ui_a2a_protocol_plugin_handles_all_orchestrator_tools(
    tool_name: str,
) -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.adk_a2a_plugin import A2uiA2AProtocolPlugin

    response = {
        "status": "draft_updated",
        "a2uiParts": [_a2ui_part(f"surface_{tool_name}", "Updated draft.")],
    }
    event = _function_response_event(tool_name, response)

    # Act
    modified_event = await A2uiA2AProtocolPlugin().on_event_callback(
        invocation_context=SimpleNamespace(),
        event=event,
    )

    # Assert
    assert modified_event is not None
    assert [part.data for part in _a2ui_data_parts_from_event(modified_event)] == [
        response["a2uiParts"][0]["data"]
    ]


@pytest.mark.asyncio
async def test_a2ui_a2a_protocol_plugin_preserves_incremental_validation_context() -> (
    None
):
    # Arrange
    from orchestrator_demo.a2ui_support.adk_a2a_plugin import A2uiA2AProtocolPlugin

    surface_id = "surface_incremental_plugin"
    response = {
        "status": "draft_updated",
        "a2uiParts": [_incremental_a2ui_part(surface_id)],
    }
    event = _function_response_event("add_plan_instruction", response)

    # Act
    modified_event = await A2uiA2AProtocolPlugin().on_event_callback(
        invocation_context=_invocation_context_with_existing_component(surface_id),
        event=event,
    )

    # Assert
    assert modified_event is not None
    assert [part.data for part in _a2ui_data_parts_from_event(modified_event)] == [
        response["a2uiParts"][0]["data"]
    ]


@pytest.mark.asyncio
async def test_a2ui_a2a_protocol_plugin_ignores_unowned_function_responses() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.adk_a2a_plugin import A2uiA2AProtocolPlugin

    response = {
        "status": "plan_required",
        "a2uiParts": [_a2ui_part("surface_unowned", "Should not bridge.")],
    }
    event = _function_response_event("unowned_tool", response)

    # Act
    modified_event = await A2uiA2AProtocolPlugin().on_event_callback(
        invocation_context=SimpleNamespace(),
        event=event,
    )

    # Assert
    assert modified_event is None
