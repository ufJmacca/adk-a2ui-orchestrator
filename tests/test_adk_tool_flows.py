from __future__ import annotations

import inspect
import json
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import pytest
from a2a import types as a2a_types
from google.adk.a2a.converters import part_converter
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.events.ui_widget import UiWidget
from google.adk.flows.llm_flows import functions as adk_functions
from google.adk.tools.base_tool import BaseTool
from google.genai import types

from orchestrator_demo.agents import build_default_specialists
from orchestrator_demo.a2a_support.transport import DataPart
from orchestrator_demo.a2ui_support.renderer_contract import (
    prepare_specialist_a2ui_for_renderer,
)
from orchestrator_demo.a2ui_support.schema_manager import A2UI_VERSION
from orchestrator_demo.contracts import (
    IntentSuggestion,
    LlmIntentAssessment,
    RoutingDecision,
)
from orchestrator_demo.orchestrator.agent import (
    ORCHESTRATOR_SESSION_STATE_KEY,
    AdkOrchestratorAdapter,
    OrchestratorAgent,
    build_root_agent,
)
from orchestrator_demo.orchestrator.service import (
    OrchestratorRequestResult,
    OrchestratorService,
    OrchestratorUserActionResult,
)
from orchestrator_demo.orchestrator.request_context import RequestContext
from orchestrator_demo.orchestrator.surface_routes import SurfaceRouteRegistry


A2UI_MIME_TYPE = "application/json+a2ui"


class FakeToolContext:
    def __init__(self) -> None:
        self.state: dict[str, Any] = {}
        self.actions = EventActions()
        self.function_call_id = "call_a2ui"
        self.saved_artifacts: list[dict[str, Any]] = []
        self.rendered_ui_widgets: list[UiWidget] = []

    async def save_artifact(
        self,
        filename: str,
        artifact: Any,
        custom_metadata: dict[str, Any] | None = None,
    ) -> int:
        version = len(
            [
                saved
                for saved in self.saved_artifacts
                if saved["filename"] == filename
            ]
        )
        self.saved_artifacts.append(
            {
                "filename": filename,
                "artifact": artifact,
                "customMetadata": custom_metadata,
                "version": version,
            }
        )
        return version

    def render_ui_widget(self, ui_widget: UiWidget) -> None:
        self.rendered_ui_widgets.append(ui_widget)


def _artifact_document(saved_artifact: dict[str, Any]) -> dict[str, Any]:
    artifact = saved_artifact["artifact"]
    assert isinstance(artifact, types.Part)
    assert artifact.text is not None
    assert artifact.inline_data is None
    return json.loads(artifact.text)


def _assert_plan_response_contract(response: dict[str, Any]) -> None:
    plan = response["plan"]
    assert response["status"] == "plan_required"
    assert response["path"] == "plan_required"
    assert response["planId"] == plan["planId"]
    assert response["planVersion"] == plan["planVersion"]
    assert response["approvalSurfaceId"] == plan["approvalSurfaceId"]
    assert response["stepIds"] == [step["stepId"] for step in plan["steps"]]
    assert response["selectedAgents"] == plan["selectedAgents"]
    assert response["stepInstructions"] == [
        {"stepId": step["stepId"], "instruction": step["instruction"]}
        for step in plan["steps"]
    ]
    assert response["dependencies"] == [
        {"stepId": step["stepId"], "dependsOn": step["dependsOn"]}
        for step in plan["steps"]
    ]
    assert {action["toolName"] for action in response["nextActions"]} == {
        "add_plan_instruction",
        "remove_plan_step",
        "replace_plan_agent",
        "reorder_plan_steps",
        "approve_orchestrator_plan",
        "reject_orchestrator_plan",
    }
    assert response["a2uiParts"]
    assert all(
        part["type"] == "data"
        and part["metadata"]["mimeType"] == "application/json+a2ui"
        for part in response["a2uiParts"]
    )


def _assert_draft_updated_contract(
    response: dict[str, Any],
    previous_response: dict[str, Any],
) -> None:
    plan = response["plan"]
    assert response["status"] == "draft_updated"
    assert response["path"] == "draft_updated"
    assert response["planId"] == previous_response["planId"]
    assert response["planId"] == plan["planId"]
    assert response["planVersion"] == previous_response["planVersion"] + 1
    assert response["planVersion"] == plan["planVersion"]
    assert response["approvalSurfaceId"] == previous_response["approvalSurfaceId"]
    assert response["approvalSurfaceId"] == plan["approvalSurfaceId"]
    assert response["stepIds"] == [step["stepId"] for step in plan["steps"]]
    assert response["a2uiParts"]
    assert all(
        part["type"] == "data"
        and part["metadata"]["mimeType"] == "application/json+a2ui"
        for part in response["a2uiParts"]
    )
    assert response["approvalResult"]["graphCreated"] is False
    assert response["approvalResult"]["specialistsCalled"] is False


def _assert_data_part_payloads(response: dict[str, Any]) -> None:
    assert response["a2uiParts"]
    assert all(
        part["type"] == "data"
        and part["metadata"]["mimeType"] == "application/json+a2ui"
        for part in response["a2uiParts"]
    )


def _assert_no_custom_renderer_transport_fields(payload: Any) -> None:
    rendered = json.dumps(payload, sort_keys=True)
    for forbidden in (
        "/api/request",
        "/api/user-action",
        "/api/status",
        "/api/status/stream",
        "/api/artifacts",
        "/static/renderer.js",
    ):
        assert forbidden not in rendered


def _adk_response_event_for(
    response: dict[str, Any],
    *,
    tool_context: Any | None = None,
) -> Any:
    tool = BaseTool(name="submit_orchestrator_request", description="test")
    if tool_context is None:
        tool_context = SimpleNamespace(
            function_call_id="call_a2ui",
            actions=EventActions(),
        )
    invocation_context = SimpleNamespace(
        invocation_id="invocation_a2ui",
        agent=SimpleNamespace(name="orchestrator"),
        branch=None,
    )
    return adk_functions.__build_response_event(
        tool,
        response,
        tool_context,
        invocation_context,
    )


def _tagged_a2ui_data_part(data: Mapping[str, Any]) -> types.Part:
    source_data_part = a2a_types.DataPart(
        data=dict(data),
        metadata={"mimeType": A2UI_MIME_TYPE},
    )
    inline_blob = (
        part_converter.A2A_DATA_PART_START_TAG
        + source_data_part.model_dump_json(
            by_alias=True,
            exclude_none=True,
        ).encode("utf-8")
        + part_converter.A2A_DATA_PART_END_TAG
    )
    return types.Part(
        inline_data=types.Blob(
            data=inline_blob,
            mime_type=part_converter.A2A_DATA_PART_TEXT_MIME_TYPE,
        )
    )


def _assert_a2ui_transport_is_filtered_from_model_history(event: Event) -> None:
    from google.adk.flows.llm_flows import contents as adk_contents

    function_call = types.Part.from_function_call(
        name="submit_orchestrator_request",
        args={"user_input": "test request"},
    )
    function_call.function_call.id = "call_a2ui"
    function_call_event = Event(
        invocation_id="invocation_a2ui",
        author="orchestrator",
        content=types.Content(role="model", parts=[function_call]),
    )

    model_contents = adk_contents._get_contents(
        None,
        [function_call_event, event],
        "orchestrator",
    )

    assert len(model_contents) == 2
    response_content = model_contents[1]
    assert response_content.parts is not None
    assert len(response_content.parts) == 1
    assert response_content.parts[0].function_response is not None
    assert response_content.parts[0].inline_data is None
    serialized = json.dumps(
        [
            content.model_dump(by_alias=True, mode="json", exclude_none=True)
            for content in model_contents
        ],
        sort_keys=True,
    )
    assert "a2a_datapart_json" not in serialized


def _a2ui_data_parts_from_event(event: Event) -> list[a2a_types.DataPart]:
    assert event.content is not None
    assert event.content.parts is not None

    return _a2ui_data_parts_from_content_parts(event.content.parts)


def _a2ui_data_parts_from_content_parts(
    content_parts: list[types.Part],
) -> list[a2a_types.DataPart]:
    data_parts: list[a2a_types.DataPart] = []
    for content_part in content_parts:
        converted_part = part_converter.convert_genai_part_to_a2a_part(content_part)
        if converted_part is None:
            continue
        data_part = converted_part.root
        assert isinstance(data_part, a2a_types.DataPart)
        if (
            isinstance(data_part.metadata, Mapping)
            and data_part.metadata.get("mimeType") == A2UI_MIME_TYPE
        ):
            data_parts.append(data_part)
    return data_parts


def _assert_a2ui_transport_events_are_filtered_from_model_history(
    events: list[Event],
) -> None:
    from google.adk.flows.llm_flows import contents as adk_contents

    function_call = types.Part.from_function_call(
        name="submit_orchestrator_request",
        args={"user_input": "test request"},
    )
    function_call.function_call.id = "call_a2ui"
    function_call_event = Event(
        invocation_id="invocation_a2ui",
        author="orchestrator",
        content=types.Content(role="model", parts=[function_call]),
    )

    model_contents = adk_contents._get_contents(
        None,
        [function_call_event, *events],
        "orchestrator",
    )

    assert len(model_contents) == 2
    response_content = model_contents[1]
    assert response_content.parts is not None
    assert [part.function_response is not None for part in response_content.parts] == [
        True
    ]
    assert all(part.inline_data is None for part in response_content.parts)
    serialized = json.dumps(
        [
            content.model_dump(by_alias=True, mode="json", exclude_none=True)
            for content in model_contents
        ],
        sort_keys=True,
    )
    assert "a2a_datapart_json" not in serialized


def test_a2ui_history_filter_preserves_inbound_user_action_data_parts() -> None:
    from google.adk.flows.llm_flows import contents as adk_contents

    user_action_data = {
        "userAction": {
            "type": "approve_plan",
            "surfaceId": "surface_plan_meeting_prep",
            "payload": {
                "planId": "plan_meeting_prep",
                "editedPlanVersion": 2,
                "approvedStepIds": ["step_relationship", "step_treasury"],
            },
        }
    }
    inbound_event = Event(
        invocation_id="invocation_inbound_a2ui",
        author="user",
        content=types.Content(
            role="user",
            parts=[_tagged_a2ui_data_part(user_action_data)],
        ),
    )

    for get_model_contents in (
        adk_contents._get_contents,
        adk_contents._get_current_turn_contents,
    ):
        model_contents = get_model_contents(
            None,
            [inbound_event],
            "orchestrator",
        )

        assert len(model_contents) == 1
        content = model_contents[0]
        assert content.parts is not None
        assert len(content.parts) == 1
        assert content.parts[0].inline_data is not None

        converted_part = part_converter.convert_genai_part_to_a2a_part(
            content.parts[0],
        )
        assert converted_part is not None
        data_part = converted_part.root
        assert isinstance(data_part, a2a_types.DataPart)
        assert data_part.metadata == {"mimeType": A2UI_MIME_TYPE}
        assert data_part.data == user_action_data


def _assert_a2ui_transport_is_wired_to_adk_content(
    response: dict[str, Any],
    *,
    tool_context: Any | None = None,
) -> dict[str, Any]:
    from orchestrator_demo.a2ui_support.adk_ui_delivery import (
        adk_content_parts_for_a2ui_response,
        adk_dev_ui_content_parts_for_a2ui_response,
    )

    event = _adk_response_event_for(response, tool_context=tool_context)
    payload = event.model_dump(by_alias=True, mode="json", exclude_none=True)
    assert event.custom_metadata == {"a2a:response": True}
    assert payload["customMetadata"] == {"a2a:response": True}
    assert event.content is not None
    assert event.content.parts is not None
    assert len(event.content.parts) > 1

    function_response = event.content.parts[0].function_response
    assert function_response is not None
    assert function_response.response == response
    assert event.content.parts[0].inline_data is None

    a2a_data_parts = _a2ui_data_parts_from_content_parts(event.content.parts[1:])
    expected_dev_ui_parts = _a2ui_data_parts_from_content_parts(
        adk_dev_ui_content_parts_for_a2ui_response(
            response,
            tool_context=tool_context,
        )
    )
    standard_a2a_parts = _a2ui_data_parts_from_content_parts(
        adk_content_parts_for_a2ui_response(
            response,
            tool_context=tool_context,
        )
    )
    assert a2a_data_parts == expected_dev_ui_parts
    assert all(
        part.metadata
        and part.metadata.get("mimeType") == "application/json+a2ui"
        for part in a2a_data_parts
    )
    expected_a2ui_data = [
        part["data"]
        for part in response["a2uiParts"]
        if isinstance(part, Mapping)
        and isinstance(part.get("metadata"), Mapping)
        and part["metadata"].get("mimeType") == A2UI_MIME_TYPE
    ]
    assert [part.data for part in standard_a2a_parts] == expected_a2ui_data
    assert all("version" in part.data for part in standard_a2a_parts)
    assert all(
        not {"beginRendering", "surfaceUpdate", "dataModelUpdate"}.intersection(
            part.data
        )
        for part in standard_a2a_parts
    )
    assert all(
        {
            "beginRendering",
            "surfaceUpdate",
            "dataModelUpdate",
            "deleteSurface",
        }.intersection(part.data)
        for part in a2a_data_parts
    )
    _assert_a2ui_transport_is_filtered_from_model_history(event)

    _assert_no_custom_renderer_transport_fields(response)
    _assert_no_custom_renderer_transport_fields(payload)
    return payload


def _assert_latest_a2ui_widget(
    tool_context: FakeToolContext,
    *,
    widget_id: str,
    response: dict[str, Any],
) -> None:
    assert tool_context.rendered_ui_widgets
    widget = tool_context.rendered_ui_widgets[-1]
    assert isinstance(widget, UiWidget)
    assert widget.id == widget_id
    assert widget.provider == "a2ui"
    assert widget.payload == {"parts": response["a2uiParts"]}
    _assert_no_custom_renderer_transport_fields(response)
    _assert_no_custom_renderer_transport_fields(widget.model_dump(mode="json"))


def _first_a2ui_surface_id(response: dict[str, Any]) -> str:
    for part in response["a2uiParts"]:
        data = part["data"]
        for message_type in (
            "createSurface",
            "updateComponents",
            "deleteSurface",
            "updateDataModel",
        ):
            message = data.get(message_type)
            if isinstance(message, dict) and isinstance(message.get("surfaceId"), str):
                return message["surfaceId"]
    raise AssertionError("response did not include an A2UI surface id")


def _assert_approval_surface_deleted(
    response: dict[str, Any],
    approval_surface_id: str,
) -> None:
    delete_parts = [
        part
        for part in response["a2uiParts"]
        if isinstance(part.get("data"), dict)
        and isinstance(part["data"].get("deleteSurface"), dict)
    ]
    assert delete_parts
    assert any(
        part["data"]["deleteSurface"]["surfaceId"] == approval_surface_id
        for part in delete_parts
    )


def _a2ui_update_components(response: dict[str, Any]) -> list[dict[str, Any]]:
    update_parts = [
        part
        for part in response["a2uiParts"]
        if isinstance(part.get("data"), dict)
        and isinstance(part["data"].get("updateComponents"), dict)
    ]
    assert len(update_parts) == 1
    update = update_parts[0]["data"]["updateComponents"]
    assert update["surfaceId"] == response["approvalSurfaceId"]
    assert isinstance(update["components"], list)
    return update["components"]


def _a2ui_component_by_id(
    components: list[dict[str, Any]],
    component_id: str,
) -> dict[str, Any]:
    matches = [component for component in components if component.get("id") == component_id]
    assert len(matches) == 1
    return matches[0]


def _a2ui_step_text(response: dict[str, Any]) -> str:
    components = _a2ui_update_components(response)
    steps = _a2ui_component_by_id(
        components,
        f"component_{response['planId']}_steps",
    )
    return steps["text"]


def _assert_a2ui_update_reflects_plan(response: dict[str, Any]) -> None:
    components = _a2ui_update_components(response)
    metadata = _a2ui_component_by_id(
        components,
        f"component_{response['planId']}_metadata",
    )
    assert f"planId: {response['planId']}" in metadata["text"]
    assert f"planVersion: {response['planVersion']}" in metadata["text"]

    steps_text = _a2ui_step_text(response)
    for index, step in enumerate(response["plan"]["steps"], start=1):
        expected_line = (
            f"{index}. {step['stepId']}: {step['agentId']} - "
            f"{step['instruction']} Expected output: {step['expectedOutput']}"
        )
        assert expected_line in steps_text

    action_payloads = [
        component["action"]["event"]["context"]["payload"]
        for component in components
        if isinstance(component.get("action"), dict)
    ]
    assert action_payloads
    for payload in action_payloads:
        assert payload["planVersion"] == response["planVersion"]
        assert payload["editedPlanVersion"] == response["planVersion"]


def _direct_request_result_with_a2ui(
    a2ui_parts: tuple[DataPart, ...],
) -> OrchestratorRequestResult:
    decision = RoutingDecision(
        path="direct",
        selected_agent="product_opportunity",
        confidence=1.0,
        reason="test direct specialist response",
    )
    context = RequestContext(
        user_input="test direct specialist request",
        slm_suggestion=IntentSuggestion(
            intent="product_opportunity",
            confidence=1.0,
        ),
        llm_assessment=LlmIntentAssessment(
            intents=["product_opportunity"],
            confidence=1.0,
            complexity="simple",
            rationale="test",
            required_agents=["product_opportunity"],
        ),
        decision=decision,
    )
    return OrchestratorRequestResult(
        path="direct",
        decision=decision,
        context=context,
        a2ui_parts=a2ui_parts,
    )


def test_root_agent_exposes_required_draft_edit_tools_with_required_fields() -> None:
    # Arrange
    root_agent = build_root_agent(
        adapter=AdkOrchestratorAdapter(),
        model="gemini-2.0-flash",
    )

    # Act
    tool_signatures = {
        tool.name: inspect.signature(tool.func) for tool in root_agent.tools
    }

    # Assert
    assert {
        "add_plan_instruction",
        "remove_plan_step",
        "replace_plan_agent",
        "reorder_plan_steps",
    } <= set(tool_signatures)
    for tool_name in {
        "add_plan_instruction",
        "remove_plan_step",
        "replace_plan_agent",
        "reorder_plan_steps",
    }:
        parameters = tool_signatures[tool_name].parameters
        for required_field in (
            "plan_id",
            "approval_surface_id",
            "edited_plan_version",
        ):
            assert required_field in parameters
            assert parameters[required_field].default is inspect.Parameter.empty

    instruction = root_agent.instruction
    assert "add_plan_instruction" in instruction
    assert "remove_plan_step" in instruction
    assert "replace_plan_agent" in instruction
    assert "reorder_plan_steps" in instruction
    assert "before approve_orchestrator_plan" in instruction


def test_approval_tools_require_final_adk_contract_fields() -> None:
    # Arrange
    root_agent = build_root_agent(
        adapter=AdkOrchestratorAdapter(),
        model="gemini-2.0-flash",
    )

    # Act
    tool_signatures = {
        tool.name: inspect.signature(tool.func) for tool in root_agent.tools
    }

    # Assert
    approve_parameters = tool_signatures["approve_orchestrator_plan"].parameters
    reject_parameters = tool_signatures["reject_orchestrator_plan"].parameters
    for required_field in (
        "plan_id",
        "approval_surface_id",
        "approved_step_ids",
        "edited_plan_version",
        "tool_context",
    ):
        assert required_field in approve_parameters
        assert approve_parameters[required_field].default is inspect.Parameter.empty
    for required_field in (
        "plan_id",
        "approval_surface_id",
        "reason",
        "tool_context",
    ):
        assert required_field in reject_parameters
        assert reject_parameters[required_field].default is inspect.Parameter.empty
    assert reject_parameters["edited_plan_version"].default is None


@pytest.mark.asyncio
async def test_tools_emit_adk_a2ui_widgets_and_top_level_a2a_parts() -> None:
    # Arrange
    plan_context = FakeToolContext()
    plan_adapter = AdkOrchestratorAdapter()
    rejection_context = FakeToolContext()
    rejection_adapter = AdkOrchestratorAdapter()
    specialist_context = FakeToolContext()

    # Act
    submitted = await plan_adapter.submit_orchestrator_request(
        "Prepare me for tomorrow's meeting with ABC Manufacturing.",
        tool_context=plan_context,
    )
    updated = await plan_adapter.add_plan_instruction(
        submitted["planId"],
        submitted["approvalSurfaceId"],
        submitted["stepIds"][0],
        "Prioritize covenant follow-up questions.",
        edited_plan_version=submitted["planVersion"],
        tool_context=plan_context,
    )
    approved = await plan_adapter.approve_orchestrator_plan(
        updated["planId"],
        updated["approvalSurfaceId"],
        updated["stepIds"],
        edited_plan_version=updated["planVersion"],
        tool_context=plan_context,
    )
    submitted_for_rejection = await rejection_adapter.submit_orchestrator_request(
        "Research this prospect and give me risks, opportunities, and talking points.",
        tool_context=rejection_context,
    )
    rejected = await rejection_adapter.reject_orchestrator_plan(
        submitted_for_rejection["planId"],
        submitted_for_rejection["approvalSurfaceId"],
        "Do not run this workflow.",
        edited_plan_version=submitted_for_rejection["planVersion"],
        tool_context=rejection_context,
    )
    specialist = await AdkOrchestratorAdapter().submit_orchestrator_request(
        "What product opportunities should I consider for a cafe business?",
        tool_context=specialist_context,
    )

    # Assert
    _assert_plan_response_contract(submitted)
    _assert_draft_updated_contract(updated, submitted)
    assert submitted["plan"]
    assert updated["plan"]
    assert submitted["nextActions"]
    assert updated["nextActions"]
    assert approved["status"] == "approved"
    assert rejected["status"] == "rejected"
    assert specialist["status"] == "direct"
    for response in (approved, rejected, specialist):
        _assert_data_part_payloads(response)

    history_events = [
        _assert_a2ui_transport_is_wired_to_adk_content(
            response,
            tool_context=context,
        )
        for response, context in (
            (submitted, plan_context),
            (updated, plan_context),
            (approved, plan_context),
            (rejected, rejection_context),
            (specialist, specialist_context),
        )
    ]

    plan_widgets = plan_context.rendered_ui_widgets
    assert len(plan_widgets) == 3
    assert [widget.id for widget in plan_widgets] == [
        submitted["approvalSurfaceId"],
        updated["approvalSurfaceId"],
        approved["approvalSurfaceId"],
    ]
    assert all(widget.provider == "a2ui" for widget in plan_widgets)
    assert plan_widgets[0].payload == {"parts": submitted["a2uiParts"]}
    assert plan_widgets[1].payload == {"parts": updated["a2uiParts"]}
    assert plan_widgets[2].payload == {"parts": approved["a2uiParts"]}

    _assert_latest_a2ui_widget(
        rejection_context,
        widget_id=rejected["approvalSurfaceId"],
        response=rejected,
    )
    _assert_latest_a2ui_widget(
        specialist_context,
        widget_id=_first_a2ui_surface_id(specialist),
        response=specialist,
    )

    for payload in (
        submitted,
        updated,
        approved,
        rejected,
        specialist,
        history_events,
        [widget.model_dump(mode="json") for widget in plan_widgets],
    ):
        _assert_no_custom_renderer_transport_fields(payload)


@pytest.mark.asyncio
async def test_adk_delivery_preserves_standard_a2a_a2ui_data_part_shape() -> None:
    # Arrange
    from google.adk.a2a.converters.from_adk_event import convert_event_to_a2a_events

    from orchestrator_demo.a2ui_support.adk_ui_delivery import (
        _a2ui_transport_response_events_for_delivery,
    )

    tool_context = FakeToolContext()
    adapter = AdkOrchestratorAdapter()
    submitted = await adapter.submit_orchestrator_request(
        "Research this prospect and give me risks, opportunities, and talking points.",
        tool_context=tool_context,
    )

    # Act
    approved = await adapter.approve_orchestrator_plan(
        submitted["planId"],
        submitted["approvalSurfaceId"],
        submitted["stepIds"],
        edited_plan_version=submitted["planVersion"],
        tool_context=tool_context,
    )
    event = _adk_response_event_for(approved, tool_context=tool_context)
    delivery_events = _a2ui_transport_response_events_for_delivery(event)

    # Assert
    assert approved["status"] == "approved"
    assert delivery_events
    assert all(
        event.custom_metadata == {"a2a:response": True}
        for event in delivery_events
    )

    exported_data = [
        data_part.data
        for delivery_event in delivery_events
        for data_part in _a2ui_data_parts_from_event(delivery_event)
    ]
    assert exported_data
    assert any(
        {
            "beginRendering",
            "surfaceUpdate",
            "dataModelUpdate",
            "deleteSurface",
        }.intersection(data)
        for data in exported_data
    )

    a2a_events: list[Any] = []
    agents_artifacts: dict[str, str] = {}
    for delivery_event in delivery_events:
        a2a_events.extend(
            convert_event_to_a2a_events(
                delivery_event,
                agents_artifacts,
                "task_a2ui",
                "context_a2ui",
            )
        )

    exported_data = [
        part.root.data
        for a2a_event in a2a_events
        if getattr(a2a_event, "artifact", None) is not None
        for part in a2a_event.artifact.parts
        if isinstance(part.root, a2a_types.DataPart)
        and isinstance(part.root.metadata, Mapping)
        and part.root.metadata.get("mimeType") == A2UI_MIME_TYPE
    ]
    expected_data = [part["data"] for part in approved["a2uiParts"]]
    assert exported_data == expected_data
    assert all("version" in data for data in exported_data)
    assert all(
        not {"beginRendering", "surfaceUpdate", "dataModelUpdate"}.intersection(data)
        for data in exported_data
    )

    surface_ids: list[str] = []
    for delivery_event in delivery_events:
        messages = [
            next(
                message
                for message in data_part.data.values()
                if isinstance(message, dict)
                and isinstance(message.get("surfaceId"), str)
            )
            for data_part in _a2ui_data_parts_from_event(delivery_event)
            if any(
                isinstance(message, dict) and isinstance(message.get("surfaceId"), str)
                for message in data_part.data.values()
            )
        ]
        surface_ids.extend(message["surfaceId"] for message in messages)

    assert submitted["approvalSurfaceId"] in surface_ids
    assert any("product_opportunity" in surface_id for surface_id in surface_ids)

    first_parts = delivery_events[0].content.parts
    assert first_parts is not None
    assert any(part.function_response is not None for part in first_parts)
    for delivery_event in delivery_events[1:]:
        assert delivery_event.content is not None
        assert delivery_event.content.parts is not None
        assert all(
            part.function_response is None
            for part in delivery_event.content.parts
        )
        assert delivery_event.actions.state_delta == {}
        assert delivery_event.actions.artifact_delta == {}
        assert delivery_event.actions.render_ui_widgets is None

    _assert_a2ui_transport_events_are_filtered_from_model_history(delivery_events)


@pytest.mark.asyncio
async def test_a2a_protocol_plugin_skips_dev_ui_a2ui_transport_events() -> None:
    # Arrange
    from google.adk.a2a.converters.from_adk_event import convert_event_to_a2a_events

    from orchestrator_demo.a2ui_support.adk_a2a_plugin import A2uiA2AProtocolPlugin

    a2ui_part = DataPart(
        data={
            "version": A2UI_VERSION,
            "updateComponents": {
                "surfaceId": "surface_dev_ui_plugin_gate",
                "components": [
                    {
                        "component": "Text",
                        "id": "root",
                        "text": "Review this draft plan.",
                    }
                ],
            },
        },
        metadata={"mimeType": A2UI_MIME_TYPE},
    ).model_dump(by_alias=True, mode="json")
    response = {
        "status": "plan_required",
        "approvalSurfaceId": "surface_dev_ui_plugin_gate",
        "a2uiParts": [a2ui_part],
    }
    event = _adk_response_event_for(response)

    # Act
    modified_event = await A2uiA2AProtocolPlugin().on_event_callback(
        invocation_context=SimpleNamespace(),
        event=event,
    )

    # Assert
    assert modified_event is None
    dev_ui_data = [part.data for part in _a2ui_data_parts_from_event(event)]
    assert len(dev_ui_data) == 1
    assert "surfaceUpdate" in dev_ui_data[0]
    assert "updateComponents" not in dev_ui_data[0]
    assert (
        dev_ui_data[0]["surfaceUpdate"]["surfaceId"]
        == "surface_dev_ui_plugin_gate"
    )

    a2a_events = convert_event_to_a2a_events(
        event,
        {},
        "task_dev_ui_plugin_gate",
        "context_dev_ui_plugin_gate",
    )
    exported_data = [
        part.root.data
        for a2a_event in a2a_events
        if getattr(a2a_event, "artifact", None) is not None
        for part in a2a_event.artifact.parts
        if isinstance(part.root, a2a_types.DataPart)
        and isinstance(part.root.metadata, Mapping)
        and part.root.metadata.get("mimeType") == A2UI_MIME_TYPE
    ]
    assert exported_data == [a2ui_part["data"]]


def test_parallel_a2ui_response_merge_preserves_marker_and_filters_history() -> None:
    # Arrange
    from google.adk.flows.llm_flows import contents as adk_contents

    a2ui_part = DataPart(
        data={
            "version": A2UI_VERSION,
            "updateComponents": {
                "surfaceId": "surface_parallel_a2ui",
                "components": [
                    {
                        "component": "Text",
                        "id": "root",
                        "text": "Parallel A2UI update.",
                    }
                ],
            },
        },
        metadata={"mimeType": A2UI_MIME_TYPE},
    ).model_dump(by_alias=True, mode="json")
    a2ui_event = _adk_response_event_for(
        {"approvalSurfaceId": "surface_parallel_a2ui", "a2uiParts": [a2ui_part]}
    )
    other_context = SimpleNamespace(
        function_call_id="call_other",
        actions=EventActions(),
    )
    other_event = _adk_response_event_for(
        {"status": "ok", "a2uiParts": []},
        tool_context=other_context,
    )

    function_call = types.Part.from_function_call(
        name="submit_orchestrator_request",
        args={"user_input": "test request"},
    )
    function_call.function_call.id = "call_a2ui"
    other_function_call = types.Part.from_function_call(
        name="submit_orchestrator_request",
        args={"user_input": "other request"},
    )
    other_function_call.function_call.id = "call_other"
    function_call_event = Event(
        invocation_id="invocation_a2ui",
        author="orchestrator",
        content=types.Content(
            role="model",
            parts=[function_call, other_function_call],
        ),
    )

    # Act
    merged_event = adk_functions.merge_parallel_function_response_events(
        [other_event, a2ui_event]
    )

    # Assert
    assert merged_event.custom_metadata == {"a2a:response": True}
    assert len(_a2ui_data_parts_from_event(merged_event)) == 1

    model_contents = adk_contents._get_contents(
        None,
        [function_call_event, merged_event],
        "orchestrator",
    )
    assert len(model_contents) == 2
    response_content = model_contents[1]
    assert response_content.parts is not None
    assert len(response_content.parts) == 2
    assert all(part.function_response is not None for part in response_content.parts)
    assert all(part.inline_data is None for part in response_content.parts)
    serialized = json.dumps(
        [
            content.model_dump(by_alias=True, mode="json", exclude_none=True)
            for content in model_contents
        ],
        sort_keys=True,
    )
    assert "a2a_datapart_json" not in serialized


@pytest.mark.parametrize(
    "response",
    [
        {"status": "ok"},
        {"status": "ok", "a2uiParts": []},
    ],
)
def test_adk_response_event_ignores_stale_a2ui_cache_without_current_parts(
    response: dict[str, Any],
) -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.adk_ui_delivery import (
        _VALIDATED_A2UI_PARTS_BY_RESPONSE_ID_ATTR,
    )

    tool_context = SimpleNamespace(
        function_call_id="call_stale_cache",
        actions=EventActions(),
    )
    stale_part = DataPart(
        data={
            "version": A2UI_VERSION,
            "updateComponents": {
                "surfaceId": "surface_stale_cache",
                "components": [
                    {
                        "component": "Text",
                        "id": "root",
                        "text": "This stale payload must not be delivered.",
                    }
                ],
            },
        },
        metadata={"mimeType": A2UI_MIME_TYPE},
    ).model_dump(by_alias=True, mode="json")
    setattr(
        tool_context,
        _VALIDATED_A2UI_PARTS_BY_RESPONSE_ID_ATTR,
        {id(response): [stale_part]},
    )

    # Act
    event = _adk_response_event_for(response, tool_context=tool_context)

    # Assert
    assert _a2ui_data_parts_from_event(event) == []
    assert event.custom_metadata != {"a2a:response": True}
    cached_parts_by_response_id = getattr(
        tool_context,
        _VALIDATED_A2UI_PARTS_BY_RESPONSE_ID_ATTR,
    )
    assert id(response) not in cached_parts_by_response_id


@pytest.mark.asyncio
async def test_adk_delivery_preserves_validated_incremental_specialist_update() -> None:
    # Arrange
    surface_id = "surface_incremental_specialist"
    registry = SurfaceRouteRegistry()
    initial_parts = tuple(
        prepare_specialist_a2ui_for_renderer(
            {
                "version": A2UI_VERSION,
                "updateComponents": {
                    "surfaceId": surface_id,
                    "components": [
                        {
                            "component": "Column",
                            "id": "root",
                            "children": ["component_prior_label"],
                        },
                        {
                            "component": "Text",
                            "id": "component_prior_label",
                            "text": "Apply recommendation",
                        },
                    ],
                },
            },
            owner_agent_id="product_opportunity",
            surface_registry=registry,
        )
    )
    incremental_parts = tuple(
        prepare_specialist_a2ui_for_renderer(
            {
                "version": A2UI_VERSION,
                "updateComponents": {
                    "surfaceId": surface_id,
                    "components": [
                        {
                            "component": "Button",
                            "id": "component_followup_button",
                            "child": "component_prior_label",
                            "action": {
                                "event": {
                                    "name": "specialist_followup",
                                    "context": {
                                        "type": "request_followup",
                                        "surfaceId": surface_id,
                                        "payload": {"source": "incremental"},
                                    },
                                }
                            },
                        }
                    ],
                },
            },
            owner_agent_id="product_opportunity",
            surface_registry=registry,
        )
    )

    class IncrementalSpecialistAgent(OrchestratorAgent):
        def __init__(self) -> None:
            self._results = [
                _direct_request_result_with_a2ui(initial_parts),
                _direct_request_result_with_a2ui(incremental_parts),
            ]
            self._session_service = OrchestratorService()

        async def handle_request(self, user_input: str) -> OrchestratorRequestResult:
            return self._results.pop(0)

        def export_session_snapshot(self) -> dict[str, Any]:
            return self._session_service.export_session_snapshot()

        def restore_session_snapshot(self, snapshot: Mapping[str, Any]) -> None:
            self._session_service.restore_session_snapshot(snapshot)

        def reset_session_snapshot(self) -> None:
            self._session_service = OrchestratorService()

        def artifact_refs(self) -> dict[str, Any]:
            return self._session_service.artifact_refs()

        def record_artifact_refs(self, artifact_refs: Mapping[str, Any]) -> None:
            self._session_service.record_artifact_refs(artifact_refs)

    tool_context = FakeToolContext()
    adapter = AdkOrchestratorAdapter(agent=IncrementalSpecialistAgent())

    # Act
    initial = await adapter.submit_orchestrator_request(
        "show the initial specialist surface",
        tool_context=tool_context,
    )
    incremental = await adapter.submit_orchestrator_request(
        "send an incremental specialist surface update",
        tool_context=tool_context,
    )

    # Assert
    assert initial["status"] == "direct"
    assert incremental["status"] == "direct"
    assert incremental["a2uiParts"] == [
        part.model_dump(by_alias=True, mode="json") for part in incremental_parts
    ]
    assert len(tool_context.rendered_ui_widgets) == 2
    widget = tool_context.rendered_ui_widgets[-1]
    assert widget.id == surface_id
    assert widget.provider == "a2ui"
    assert widget.payload == {"parts": incremental["a2uiParts"]}
    assert (
        widget.payload["parts"][0]["data"]["updateComponents"]["components"][0]["child"]
        == "component_prior_label"
    )
    assert incremental.get("error", {}).get("code") != "a2ui_delivery_error"

    event = _adk_response_event_for(incremental, tool_context=tool_context)
    assert event.content is not None
    assert event.content.parts is not None
    assert len(event.content.parts) == 2
    converted_part = part_converter.convert_genai_part_to_a2a_part(
        event.content.parts[1]
    )
    assert converted_part is not None
    data_part = converted_part.root
    assert isinstance(data_part, a2a_types.DataPart)
    assert data_part.metadata
    assert data_part.metadata["mimeType"] == "application/json+a2ui"
    assert "surfaceUpdate" in data_part.data
    assert "updateComponents" not in data_part.data
    button = data_part.data["surfaceUpdate"]["components"][0]
    assert button["component"]["Button"]["child"] == "component_prior_label"

    from google.adk.a2a.converters.from_adk_event import convert_event_to_a2a_events

    a2a_events = convert_event_to_a2a_events(
        event,
        {},
        "task_incremental_a2ui",
        "context_incremental_a2ui",
    )
    exported_data = [
        part.root.data
        for a2a_event in a2a_events
        if getattr(a2a_event, "artifact", None) is not None
        for part in a2a_event.artifact.parts
        if isinstance(part.root, a2a_types.DataPart)
        and isinstance(part.root.metadata, Mapping)
        and part.root.metadata.get("mimeType") == A2UI_MIME_TYPE
    ]
    assert exported_data == [incremental["a2uiParts"][0]["data"]]


@pytest.mark.asyncio
async def test_draft_edit_tools_synthesize_existing_user_action_envelopes() -> None:
    # Arrange
    class RecordingAgent(OrchestratorAgent):
        def __init__(self) -> None:
            self.user_actions: list[dict[str, Any]] = []

        async def handle_user_action(
            self,
            user_action: Any,
        ) -> OrchestratorUserActionResult:
            self.user_actions.append(user_action)
            return OrchestratorUserActionResult(status="ignored")

    recording_agent = RecordingAgent()
    adapter = AdkOrchestratorAdapter(agent=recording_agent)

    # Act
    await adapter.add_plan_instruction(
        "plan_example",
        "surface_plan_example",
        "step_relationship_summary",
        "Prioritize recent deposit trends.",
        edited_plan_version=7,
    )
    await adapter.remove_plan_step(
        "plan_example",
        "surface_plan_example",
        "step_industry_research",
        edited_plan_version=8,
    )
    await adapter.replace_plan_agent(
        "plan_example",
        "surface_plan_example",
        "step_industry_research",
        "credit_risk",
        edited_plan_version=9,
    )
    await adapter.reorder_plan_steps(
        "plan_example",
        "surface_plan_example",
        ["step_internal_knowledge", "step_relationship_summary"],
        edited_plan_version=10,
    )

    # Assert
    assert recording_agent.user_actions == [
        {
            "userAction": {
                "type": "add_instruction",
                "surfaceId": "surface_plan_example",
                "payload": {
                    "planId": "plan_example",
                    "editedPlanVersion": 7,
                    "stepId": "step_relationship_summary",
                    "instruction": "Prioritize recent deposit trends.",
                },
            }
        },
        {
            "userAction": {
                "type": "remove_step",
                "surfaceId": "surface_plan_example",
                "payload": {
                    "planId": "plan_example",
                    "editedPlanVersion": 8,
                    "stepId": "step_industry_research",
                },
            }
        },
        {
            "userAction": {
                "type": "replace_agent",
                "surfaceId": "surface_plan_example",
                "payload": {
                    "planId": "plan_example",
                    "editedPlanVersion": 9,
                    "stepId": "step_industry_research",
                    "replacementAgentId": "credit_risk",
                },
            }
        },
        {
            "userAction": {
                "type": "reorder_steps",
                "surfaceId": "surface_plan_example",
                "payload": {
                    "planId": "plan_example",
                    "editedPlanVersion": 10,
                    "orderedStepIds": [
                        "step_internal_knowledge",
                        "step_relationship_summary",
                    ],
                },
            }
        },
    ]


@pytest.mark.asyncio
async def test_draft_edit_tools_update_plan_a2ui_and_session_without_execution() -> None:
    # Arrange
    tool_context = FakeToolContext()
    service = OrchestratorService()
    adapter = AdkOrchestratorAdapter(agent=OrchestratorAgent(service))
    submitted = await adapter.submit_orchestrator_request(
        "Prepare me for tomorrow's meeting with ABC Manufacturing.",
        tool_context=tool_context,
    )

    # Act
    instructed = await adapter.add_plan_instruction(
        submitted["planId"],
        submitted["approvalSurfaceId"],
        "step_internal_knowledge",
        "Prioritize covenant follow-up questions.",
        edited_plan_version=submitted["planVersion"],
        tool_context=tool_context,
    )
    removed = await adapter.remove_plan_step(
        instructed["planId"],
        instructed["approvalSurfaceId"],
        "step_industry_research",
        edited_plan_version=instructed["planVersion"],
        tool_context=tool_context,
    )
    replaced = await adapter.replace_plan_agent(
        removed["planId"],
        removed["approvalSurfaceId"],
        "step_relationship_summary",
        "credit_risk",
        edited_plan_version=removed["planVersion"],
        tool_context=tool_context,
    )
    reordered = await adapter.reorder_plan_steps(
        replaced["planId"],
        replaced["approvalSurfaceId"],
        [
            "step_internal_knowledge",
            "step_relationship_summary",
            "step_synthesis",
        ],
        edited_plan_version=replaced["planVersion"],
        tool_context=tool_context,
    )

    # Assert
    for current, previous in (
        (instructed, submitted),
        (removed, instructed),
        (replaced, removed),
        (reordered, replaced),
    ):
        _assert_draft_updated_contract(current, previous)
        _assert_a2ui_update_reflects_plan(current)

    assert "Additional instruction: Prioritize covenant follow-up questions." in (
        _a2ui_step_text(instructed)
    )
    assert "step_industry_research" in _a2ui_step_text(instructed)
    assert "step_industry_research" not in _a2ui_step_text(removed)
    assert "step_relationship_summary: credit_risk" in _a2ui_step_text(replaced)
    assert _a2ui_step_text(reordered).index("1. step_internal_knowledge") < (
        _a2ui_step_text(reordered).index("2. step_relationship_summary")
    )

    assert service.specialist_call_counts() == {}
    assert tool_context.saved_artifacts == []
    snapshot = tool_context.state[ORCHESTRATOR_SESSION_STATE_KEY]
    draft = snapshot["approvalRecords"][submitted["planId"]]["draftPlan"]
    assert draft["plan_version"] == reordered["planVersion"]
    assert [step["step_id"] for step in draft["steps"]] == reordered["stepIds"]
    assert (
        "Additional instruction: Prioritize covenant follow-up questions."
        in reordered["plan"]["steps"][0]["instruction"]
    )
    assert reordered["plan"]["steps"][1]["agentId"] == "credit_risk"


@pytest.mark.asyncio
async def test_adk_context_list_edit_resolves_sibling_form_data_values() -> None:
    # Arrange
    service = OrchestratorService()
    submitted = await service.handle_user_request(
        "Prepare me for tomorrow's meeting with ABC Manufacturing."
    )
    plan = submitted.approval_plan
    assert plan is not None
    step_id = plan.steps[0].step_id
    instruction = "Prioritize covenant follow-up questions from the rendered field."

    # Act
    edited = await service.handle_user_action(
        {
            "userAction": {
                "name": "add_instruction",
                "surfaceId": plan.approval_surface_id,
                "context": [
                    {"key": "type", "value": {"literalString": "add_instruction"}},
                    {"key": "planId", "value": {"literalString": plan.plan_id}},
                    {"key": "planVersion", "value": {"literalNumber": plan.plan_version}},
                    {"key": "stepId", "value": {"literalString": step_id}},
                    {
                        "key": "instruction",
                        "value": {
                            "path": f"/approvalEdits/{step_id}/instruction",
                        },
                    },
                ],
            },
            "formData": {
                "approvalEdits": {
                    step_id: {
                        "instruction": instruction,
                    },
                },
            },
        }
    )

    # Assert
    assert edited.status == "draft_updated"
    assert edited.approval_result is not None
    updated_plan = edited.approval_result.draft_plan
    assert updated_plan is not None
    updated_step = next(step for step in updated_plan.steps if step.step_id == step_id)
    assert f"Additional instruction: {instruction}" in updated_step.instruction


@pytest.mark.asyncio
async def test_nested_direct_user_action_resolves_sibling_form_data_values() -> None:
    # Arrange
    service = OrchestratorService()
    submitted = await service.handle_user_request(
        "Prepare me for tomorrow's meeting with ABC Manufacturing."
    )
    plan = submitted.approval_plan
    assert plan is not None
    step_id = plan.steps[0].step_id
    instruction = "Prioritize covenant follow-up questions from a direct click."

    # Act
    edited = await service.handle_user_action(
        {
            "userAction": {
                "type": "add_instruction",
                "surfaceId": plan.approval_surface_id,
                "planId": plan.plan_id,
                "planVersion": plan.plan_version,
                "stepId": step_id,
                "instruction": {
                    "path": f"/approvalEdits/{step_id}/instruction",
                },
            },
            "formData": {
                "approvalEdits": {
                    step_id: {
                        "instruction": instruction,
                    },
                },
            },
        }
    )

    # Assert
    assert edited.status == "draft_updated"
    assert edited.approval_result is not None
    updated_plan = edited.approval_result.draft_plan
    assert updated_plan is not None
    updated_step = next(step for step in updated_plan.steps if step.step_id == step_id)
    assert f"Additional instruction: {instruction}" in updated_step.instruction


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("envelope_shape", "action_type", "field_name", "field_value"),
    [
        (
            "event_action",
            "add_instruction",
            "instruction",
            "Prioritize covenant follow-up questions from the nested event field.",
        ),
        ("action_event", "replace_agent", "replacementAgentId", "credit_risk"),
    ],
)
async def test_nested_adk_context_list_edit_resolves_sibling_form_data_values(
    envelope_shape: str,
    action_type: str,
    field_name: str,
    field_value: str,
) -> None:
    # Arrange
    service = OrchestratorService()
    submitted = await service.handle_user_request(
        "Prepare me for tomorrow's meeting with ABC Manufacturing."
    )
    plan = submitted.approval_plan
    assert plan is not None
    step_id = (
        plan.steps[0].step_id
        if action_type == "add_instruction"
        else "step_relationship_summary"
    )
    context = [
        {"key": "type", "value": {"literalString": action_type}},
        {"key": "surfaceId", "value": {"literalString": plan.approval_surface_id}},
        {"key": "planId", "value": {"literalString": plan.plan_id}},
        {"key": "planVersion", "value": {"literalNumber": plan.plan_version}},
        {"key": "stepId", "value": {"literalString": step_id}},
        {
            "key": field_name,
            "value": {"path": f"/approvalEdits/{step_id}/{field_name}"},
        },
    ]
    action = {"name": action_type, "context": context}
    event_payload = (
        {"event": {"action": action}}
        if envelope_shape == "event_action"
        else {"action": {"event": action}}
    )

    # Act
    edited = await service.handle_user_action(
        {
            **event_payload,
            "formData": {
                "approvalEdits": {
                    step_id: {
                        field_name: field_value,
                    },
                },
            },
        }
    )

    # Assert
    assert edited.status == "draft_updated"
    assert edited.approval_result is not None
    updated_plan = edited.approval_result.draft_plan
    assert updated_plan is not None
    updated_step = next(step for step in updated_plan.steps if step.step_id == step_id)
    if action_type == "add_instruction":
        assert f"Additional instruction: {field_value}" in updated_step.instruction
    else:
        assert updated_step.agent_id == field_value


@pytest.mark.asyncio
async def test_event_payload_mapping_preserves_json_looking_plan_text_values() -> None:
    # Arrange
    service = OrchestratorService()
    submitted = await service.handle_user_request(
        "Prepare me for tomorrow's meeting with ABC Manufacturing."
    )
    plan = submitted.approval_plan
    assert plan is not None
    step_id = plan.steps[0].step_id
    instruction = '{"priority":"high"}'
    rejection_reason = '{"reason":"narrow scope"}'

    # Act
    edited = await service.handle_user_action(
        {
            "event": {
                "name": "add_instruction",
                "context": {
                    "surfaceId": plan.approval_surface_id,
                    "payload": {
                        "planId": plan.plan_id,
                        "planVersion": plan.plan_version,
                        "stepId": step_id,
                        "instruction": instruction,
                    },
                },
            },
        }
    )
    assert edited.approval_result is not None
    edited_plan = edited.approval_result.draft_plan
    assert edited_plan is not None

    rejected = await service.handle_user_action(
        {
            "event": {
                "name": "reject_plan",
                "context": {
                    "surfaceId": plan.approval_surface_id,
                    "payload": {
                        "planId": plan.plan_id,
                        "planVersion": edited_plan.plan_version,
                        "reason": rejection_reason,
                    },
                },
            },
        }
    )

    # Assert
    assert edited.status == "draft_updated"
    updated_step = next(step for step in edited_plan.steps if step.step_id == step_id)
    assert f"Additional instruction: {instruction}" in updated_step.instruction
    assert rejected.status == "rejected"
    assert rejected.approval_result is not None
    assert rejected.approval_result.rejection_reason == rejection_reason
    assert service.specialist_call_counts() == {}


@pytest.mark.asyncio
async def test_approve_tool_executes_graph_saves_artifacts_and_persists_session() -> None:
    # Arrange
    tool_context = FakeToolContext()
    service = OrchestratorService()
    adapter = AdkOrchestratorAdapter(agent=OrchestratorAgent(service))
    submitted = await adapter.submit_orchestrator_request(
        "Prepare me for tomorrow's meeting with ABC Manufacturing.",
        tool_context=tool_context,
    )

    # Act
    approved = await adapter.approve_orchestrator_plan(
        submitted["planId"],
        submitted["approvalSurfaceId"],
        submitted["stepIds"],
        edited_plan_version=submitted["planVersion"],
        tool_context=tool_context,
    )

    # Assert
    assert approved["status"] == "approved"
    assert approved["path"] == "approved"
    assert approved["planId"] == submitted["planId"]
    assert approved["planVersion"] == submitted["planVersion"]
    assert approved["approvalSurfaceId"] == submitted["approvalSurfaceId"]
    assert approved["graphCreated"] is True
    assert approved["specialistsCalled"] is True
    assert approved["approvalResult"]["graphCreated"] is True
    assert approved["approvalResult"]["specialistsCalled"] is True
    assert approved["statusEvents"]
    assert approved["artifacts"]["final_response"]["agent_id"] == "synthesis"
    _assert_data_part_payloads(approved)
    _assert_approval_surface_deleted(approved, submitted["approvalSurfaceId"])

    latest_filename = "orchestrator_latest_result.json"
    plan_filename = f"orchestrator_plan_{submitted['planId']}_execution.json"
    assert [artifact["filename"] for artifact in tool_context.saved_artifacts] == [
        latest_filename,
        plan_filename,
    ]
    snapshot = tool_context.state[ORCHESTRATOR_SESSION_STATE_KEY]
    assert snapshot["approvalRecords"][submitted["planId"]]["status"] == "approved"
    assert set(snapshot["artifactRefs"]) == {latest_filename, plan_filename}
    assert service.specialist_call_counts() == {
        "relationship_summary": 1,
        "internal_knowledge": 1,
        "industry_research": 1,
        "synthesis": 1,
    }


@pytest.mark.asyncio
async def test_reject_tool_returns_closed_a2ui_update_and_persists_without_execution() -> None:
    # Arrange
    tool_context = FakeToolContext()
    service = OrchestratorService()
    adapter = AdkOrchestratorAdapter(agent=OrchestratorAgent(service))
    submitted = await adapter.submit_orchestrator_request(
        "Research this prospect and give me risks, opportunities, and talking points.",
        tool_context=tool_context,
    )

    # Act
    rejected = await adapter.reject_orchestrator_plan(
        submitted["planId"],
        submitted["approvalSurfaceId"],
        "Do not run this workflow.",
        edited_plan_version=submitted["planVersion"],
        tool_context=tool_context,
    )

    # Assert
    assert rejected["status"] == "rejected"
    assert rejected["path"] == "rejected"
    assert rejected["planId"] == submitted["planId"]
    assert rejected["planVersion"] == submitted["planVersion"]
    assert rejected["reason"] == "Do not run this workflow."
    assert rejected["graphCreated"] is False
    assert rejected["specialistsCalled"] is False
    assert rejected["approvalResult"]["graphCreated"] is False
    assert rejected["approvalResult"]["specialistsCalled"] is False
    assert rejected["approvalResult"]["reason"] == "Do not run this workflow."
    _assert_data_part_payloads(rejected)
    _assert_approval_surface_deleted(rejected, submitted["approvalSurfaceId"])

    assert service.specialist_call_counts() == {}
    assert tool_context.saved_artifacts == []
    snapshot = tool_context.state[ORCHESTRATOR_SESSION_STATE_KEY]
    record = snapshot["approvalRecords"][submitted["planId"]]
    assert record["status"] == "rejected"
    assert record["rejectionReason"] == "Do not run this workflow."


@pytest.mark.asyncio
async def test_approval_and_rejection_validation_failures_return_safe_errors() -> None:
    # Arrange
    adapter = AdkOrchestratorAdapter()
    tool_context = FakeToolContext()
    submitted = await adapter.submit_orchestrator_request(
        "Prepare me for tomorrow's meeting with ABC Manufacturing.",
        tool_context=tool_context,
    )
    stale_context = FakeToolContext()
    stale_context.state[ORCHESTRATOR_SESSION_STATE_KEY] = json.loads(
        json.dumps(tool_context.state[ORCHESTRATOR_SESSION_STATE_KEY])
    )
    unknown_plan_context = FakeToolContext()
    unknown_plan_context.state[ORCHESTRATOR_SESSION_STATE_KEY] = json.loads(
        json.dumps(tool_context.state[ORCHESTRATOR_SESSION_STATE_KEY])
    )
    wrong_surface_context = FakeToolContext()
    wrong_surface_snapshot = json.loads(
        json.dumps(tool_context.state[ORCHESTRATOR_SESSION_STATE_KEY])
    )
    wrong_surface_snapshot["surfaceRegistry"]["ownersBySurfaceId"][
        "surface_plan_wrong"
    ] = {
        "surfaceId": "surface_plan_wrong",
        "ownerType": "orchestrator",
        "ownerId": "orchestrator",
        "planId": submitted["planId"],
        "source": "approval_surface",
    }
    wrong_surface_context.state[ORCHESTRATOR_SESSION_STATE_KEY] = (
        wrong_surface_snapshot
    )

    updated = await adapter.add_plan_instruction(
        submitted["planId"],
        submitted["approvalSurfaceId"],
        submitted["stepIds"][0],
        "Prioritize covenant follow-up questions.",
        edited_plan_version=submitted["planVersion"],
        tool_context=tool_context,
    )
    final_context = FakeToolContext()
    final_context.state[ORCHESTRATOR_SESSION_STATE_KEY] = json.loads(
        json.dumps(tool_context.state[ORCHESTRATOR_SESSION_STATE_KEY])
    )

    specialists = build_default_specialists()
    unavailable_context = FakeToolContext()
    unavailable_context.state[ORCHESTRATOR_SESSION_STATE_KEY] = json.loads(
        json.dumps(tool_context.state[ORCHESTRATOR_SESSION_STATE_KEY])
    )
    unavailable_adapter = AdkOrchestratorAdapter(
        agent=OrchestratorAgent(
            OrchestratorService(
                specialists={
                    agent_id: specialist
                    for agent_id, specialist in specialists.items()
                    if agent_id != "synthesis"
                }
            )
        )
    )

    # Act
    stale_approval = await adapter.approve_orchestrator_plan(
        submitted["planId"],
        submitted["approvalSurfaceId"],
        submitted["stepIds"],
        edited_plan_version=submitted["planVersion"],
        tool_context=stale_context,
    )
    mismatched_steps = await adapter.approve_orchestrator_plan(
        updated["planId"],
        updated["approvalSurfaceId"],
        ["step_does_not_match_current_plan"],
        edited_plan_version=updated["planVersion"],
        tool_context=tool_context,
    )
    wrong_surface = await adapter.approve_orchestrator_plan(
        updated["planId"],
        "surface_plan_wrong",
        updated["stepIds"],
        edited_plan_version=updated["planVersion"],
        tool_context=wrong_surface_context,
    )
    unknown_plan = await adapter.approve_orchestrator_plan(
        "plan_missing",
        updated["approvalSurfaceId"],
        updated["stepIds"],
        edited_plan_version=updated["planVersion"],
        tool_context=unknown_plan_context,
    )
    unavailable_specialist = await unavailable_adapter.approve_orchestrator_plan(
        updated["planId"],
        updated["approvalSurfaceId"],
        updated["stepIds"],
        edited_plan_version=updated["planVersion"],
        tool_context=unavailable_context,
    )
    approved = await adapter.approve_orchestrator_plan(
        updated["planId"],
        updated["approvalSurfaceId"],
        updated["stepIds"],
        edited_plan_version=updated["planVersion"],
        tool_context=final_context,
    )
    final_plan_rejection = await adapter.reject_orchestrator_plan(
        updated["planId"],
        updated["approvalSurfaceId"],
        "Cannot reject a final plan.",
        edited_plan_version=updated["planVersion"],
        tool_context=final_context,
    )

    # Assert
    assert approved["status"] == "approved"
    assert {
        stale_approval["error"]["code"],
        mismatched_steps["error"]["code"],
        wrong_surface["error"]["code"],
        unknown_plan["error"]["code"],
        unavailable_specialist["error"]["code"],
        final_plan_rejection["error"]["code"],
    } == {
        "stale_plan_version",
        "invalid_plan_mutation",
        "surface_mismatch",
        "plan_not_found",
        "plan_already_final",
    }
    assert unavailable_specialist["error"]["code"] == "invalid_plan_mutation"
    for response in (
        stale_approval,
        mismatched_steps,
        wrong_surface,
        unknown_plan,
        unavailable_specialist,
        final_plan_rejection,
    ):
        assert response["status"] == "error"
        assert response["path"] == "error"
        rendered = json.dumps(response, sort_keys=True)
        assert "Traceback" not in rendered
        assert "step_does_not_match_current_plan" not in rendered
        assert "sk-" not in rendered


@pytest.mark.asyncio
async def test_failed_draft_edit_preserves_prior_draft_and_returns_safe_error() -> None:
    # Arrange
    tool_context = FakeToolContext()
    adapter = AdkOrchestratorAdapter()
    submitted = await adapter.submit_orchestrator_request(
        "Research this prospect and give me risks, opportunities, and talking points.",
        tool_context=tool_context,
    )
    draft_before_failure = json.loads(
        json.dumps(
            tool_context.state[ORCHESTRATOR_SESSION_STATE_KEY]["approvalRecords"][
                submitted["planId"]
            ]["draftPlan"]
        )
    )
    leaked_replacement_agent = (
        "OPENROUTER_API_KEY=sk-or-v1-replacement-agent-secret-should-not-leak"
    )

    # Act
    failed = await adapter.replace_plan_agent(
        submitted["planId"],
        submitted["approvalSurfaceId"],
        submitted["stepIds"][0],
        leaked_replacement_agent,
        edited_plan_version=submitted["planVersion"],
        tool_context=tool_context,
    )

    # Assert
    assert failed["status"] == "error"
    assert failed["path"] == "error"
    assert failed["error"]["code"] == "invalid_plan_mutation"
    rendered_error = json.dumps(failed, sort_keys=True)
    assert leaked_replacement_agent not in rendered_error
    assert "OPENROUTER_API_KEY" not in rendered_error
    assert "sk-or-v1-replacement-agent-secret-should-not-leak" not in rendered_error
    assert tool_context.saved_artifacts == []
    draft_after_failure = tool_context.state[ORCHESTRATOR_SESSION_STATE_KEY][
        "approvalRecords"
    ][submitted["planId"]]["draftPlan"]
    assert draft_after_failure == draft_before_failure


@pytest.mark.asyncio
async def test_submit_complex_request_restores_session_and_returns_plan_without_specialists() -> None:
    # Arrange
    first_context = FakeToolContext()
    first_service = OrchestratorService()
    first_adapter = AdkOrchestratorAdapter(agent=OrchestratorAgent(first_service))
    first_response = await first_adapter.submit_orchestrator_request(
        "Prepare me for tomorrow's meeting with ABC Manufacturing.",
        tool_context=first_context,
    )
    carried_snapshot = json.loads(
        json.dumps(first_context.state[ORCHESTRATOR_SESSION_STATE_KEY])
    )

    fresh_context = FakeToolContext()
    fresh_context.state[ORCHESTRATOR_SESSION_STATE_KEY] = carried_snapshot
    fresh_service = OrchestratorService()
    fresh_adapter = AdkOrchestratorAdapter(agent=OrchestratorAgent(fresh_service))

    # Act
    second_response = await fresh_adapter.submit_orchestrator_request(
        "Research this prospect and give me risks, opportunities, and talking points.",
        tool_context=fresh_context,
    )

    # Assert
    _assert_plan_response_contract(first_response)
    _assert_plan_response_contract(second_response)
    assert first_service.specialist_call_counts() == {}
    assert fresh_service.specialist_call_counts() == {}
    assert first_context.saved_artifacts == []
    assert fresh_context.saved_artifacts == []
    assert first_context.actions.skip_summarization is True
    assert fresh_context.actions.skip_summarization is True

    snapshot = fresh_context.state[ORCHESTRATOR_SESSION_STATE_KEY]
    assert set(snapshot["approvalRecords"]) == {
        first_response["planId"],
        second_response["planId"],
    }
    assert snapshot["approvalRecords"][first_response["planId"]]["status"] == "draft"
    assert snapshot["approvalRecords"][second_response["planId"]]["status"] == "draft"


@pytest.mark.asyncio
async def test_submit_direct_request_saves_latest_artifact_and_skips_summarization() -> None:
    # Arrange
    tool_context = FakeToolContext()

    # Act
    response = await AdkOrchestratorAdapter().submit_orchestrator_request(
        "Summarize the internal notes for ABC Manufacturing.",
        tool_context=tool_context,
    )

    # Assert
    assert response["status"] == "direct"
    assert response["path"] == "direct"
    assert tool_context.actions.skip_summarization is True
    assert [artifact["filename"] for artifact in tool_context.saved_artifacts] == [
        "orchestrator_latest_result.json"
    ]

    saved = tool_context.saved_artifacts[0]
    document = _artifact_document(saved)
    expected_ref = {
        "filename": "orchestrator_latest_result.json",
        "version": saved["version"],
        "mimeType": "application/json",
        "documentType": "direct_result",
    }
    assert saved["customMetadata"] == {
        "documentType": "direct_result",
        "mimeType": "application/json",
    }
    assert document["status"] == "direct"
    assert document["path"] == "direct"
    assert document["artifacts"]["final_response"]["agent_id"] == (
        response["artifacts"]["final_response"]["agent_id"]
    )
    assert response["artifactRefs"]["orchestrator_latest_result.json"] == expected_ref
    assert tool_context.state[ORCHESTRATOR_SESSION_STATE_KEY]["artifactRefs"][
        "orchestrator_latest_result.json"
    ] == expected_ref


@pytest.mark.parametrize(
    ("deterministic_model", "adk_eval_mode", "expected_skip_summarization"),
    [
        pytest.param(False, False, True, id="normal-runtime"),
        pytest.param(True, False, True, id="deterministic-runtime"),
        pytest.param(False, True, True, id="eval-with-live-model-runtime"),
        pytest.param(True, True, False, id="deterministic-eval-runtime"),
    ],
)
@pytest.mark.asyncio
async def test_submit_direct_request_enables_summarization_only_for_deterministic_eval(
    monkeypatch: pytest.MonkeyPatch,
    deterministic_model: bool,
    adk_eval_mode: bool,
    expected_skip_summarization: bool,
) -> None:
    # Arrange
    if deterministic_model:
        monkeypatch.setenv("ORCHESTRATOR_DEMO_DETERMINISTIC_MODEL", "1")
    else:
        monkeypatch.delenv("ORCHESTRATOR_DEMO_DETERMINISTIC_MODEL", raising=False)
    if adk_eval_mode:
        monkeypatch.setenv("ORCHESTRATOR_DEMO_ADK_EVAL_MODE", "1")
    else:
        monkeypatch.delenv("ORCHESTRATOR_DEMO_ADK_EVAL_MODE", raising=False)

    tool_context = FakeToolContext()

    # Act
    response = await AdkOrchestratorAdapter().submit_orchestrator_request(
        "Summarize the internal notes for ABC Manufacturing.",
        tool_context=tool_context,
    )

    # Assert
    assert response["status"] == "direct"
    assert response["path"] == "direct"
    assert tool_context.actions.skip_summarization is expected_skip_summarization


@pytest.mark.asyncio
async def test_plan_and_reject_tools_enable_summarization_for_deterministic_eval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    monkeypatch.setenv("ORCHESTRATOR_DEMO_DETERMINISTIC_MODEL", "1")
    monkeypatch.setenv("ORCHESTRATOR_DEMO_ADK_EVAL_MODE", "1")
    adapter = AdkOrchestratorAdapter()
    tool_context = FakeToolContext()

    # Act
    submitted = await adapter.submit_orchestrator_request(
        "Prepare me for tomorrow's meeting with ABC Manufacturing.",
        tool_context=tool_context,
    )
    plan_skip_summarization = tool_context.actions.skip_summarization
    rejected = await adapter.reject_orchestrator_plan(
        submitted["planId"],
        submitted["approvalSurfaceId"],
        "Too broad for today.",
        edited_plan_version=submitted["planVersion"],
        tool_context=tool_context,
    )

    # Assert
    assert submitted["status"] == "plan_required"
    assert rejected["status"] == "rejected"
    assert plan_skip_summarization is False
    assert tool_context.actions.skip_summarization is False


@pytest.mark.asyncio
async def test_submit_error_response_skips_summarization() -> None:
    # Arrange
    tool_context = FakeToolContext()
    tool_context.state[ORCHESTRATOR_SESSION_STATE_KEY] = ["not", "a", "snapshot"]

    # Act
    response = await AdkOrchestratorAdapter().submit_orchestrator_request(
        "Summarize the internal notes for ABC Manufacturing.",
        tool_context=tool_context,
    )

    # Assert
    assert response["status"] == "error"
    assert response["path"] == "error"
    assert response["error"]["code"] == "unexpected_error"
    assert tool_context.actions.skip_summarization is True
    assert tool_context.saved_artifacts == []
